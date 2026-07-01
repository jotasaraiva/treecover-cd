from rasterio.warp import reproject, Resampling
from shapely.geometry import box
from pathlib import Path
import geopandas as gpd
import rasterio as rio
import numpy as np
import geemap
import ee
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/pipeline.log", mode="a")
    ]
)
logger = logging.getLogger(__name__)

ee.Authenticate(auth_mode="notebook")
ee.Initialize()

class DataPipeline:

    def __init__(self) -> None:
        self.year_start = 2018
        self.year_end = 2024

    def raw(self):
    
        data = Path("data/labels/sample_2000_2024.tif")
        data.parent.mkdir(parents=True, exist_ok=True)
    
        with rio.open(data) as src:
            meta = src.meta
            bbox = src.bounds
    
        gdf = gpd.GeoDataFrame({"geometry": [box(*bbox)]}, crs=meta["crs"])
        fc = geemap.gdf_to_ee(gdf)
        geom = fc.geometry()
    
        s1 = (
            ee.ImageCollection('COPERNICUS/S1_GRD')
                .filterBounds(geom)
                .filterDate(f"{self.year_start}-01-01", f"{self.year_end + 1}-01-01")
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
                .filter(ee.Filter.eq('instrumentMode', 'IW'))
        )
    
        s1 = s1.map(lambda img: img.clip(geom))
    
        v_emit_asc = s1.filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))
    
        raw_dir = Path("data/raw/")
        os.makedirs(raw_dir, exist_ok=True)
    
        collection = v_emit_asc.select(["VV", "VH"])
    
        monthly = []
        for year in range(self.year_start, self.year_end + 1):
            for month in range(1, 13):
            
                start = ee.Date.fromYMD(year, month, 1)
                end = start.advance(1, "month")
    
                img = (
                    collection
                    .filterDate(start, end)
                    .median()
                    .clip(geom)
                    .set({
                        "system:index": f"{year}-{month:02d}",
                        "system:time_start": start.millis()
                    })
                )
    
                monthly.append(img)
    
        monthly_collection = ee.ImageCollection(monthly)
    
        geemap.ee_export_image_collection(
            monthly_collection,
            out_dir=raw_dir,
            scale=10,
            region=geom,
            file_per_band=True
        )

    def preprocessed(self):

        raw_dir = Path("data/raw/")
        out_dir = Path("data/preprocessed/")

        if not raw_dir.exists():
            raise ValueError("Run raw() first")

        out_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(list(raw_dir.glob("*.tif")))
        ref_path = files[0]

        with rio.open(ref_path) as ref:
            ref_transform = ref.transform
            ref_crs = ref.crs
            ref_shape = (ref.height, ref.width)
            ref_profile = ref.profile.copy()

        def align(src_array, src_transform, src_crs):

            dst = np.zeros(ref_shape, dtype=np.float32)

            reproject(
                source=src_array,
                destination=dst,
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.bilinear
            )

            return dst

        for f in files:

            logger.info(f"Preprocessing file {f.stem} ...")
            with rio.open(f) as src:
                data = src.read(1).astype(np.float32)

                aligned = align(
                    data,
                    src.transform,
                    src.crs
                )

            out_path = out_dir / f.name

            profile = ref_profile.copy()
            profile.update({
                "height": ref_shape[0],
                "width": ref_shape[1],
                "transform": ref_transform,
                "crs": ref_crs,
                "dtype": "float32",
                "count": 1
            })

            with rio.open(out_path, "w", **profile) as dst:
                dst.write(aligned, 1)
                
    def labels(self):
        
        def normalize(arr):
            arr = np.asarray(arr, dtype=np.float64) # Ensure float output to prevent integer division

            # Calculate min and max values
            arr_min = np.nanmin(arr)
            arr_max = np.nanmax(arr)

            # Calculate denominator and handle potential division-by-zero
            denominator = arr_max - arr_min
            denominator = np.where(denominator == 0, 1.0, denominator)

            # Apply transformation
            normalized = (arr - arr_min) / denominator

            return normalized
        
        def align_labels_to_reference(
            label_path: str | Path,
            reference_path: str | Path,
            output_path: str | Path | None = None,
        ):

            with rio.open(label_path) as src_lbl, rio.open(reference_path) as src_ref:
            
                labels = src_lbl.read(1)

                aligned = np.zeros(
                    (src_ref.height, src_ref.width),
                    dtype=src_lbl.dtypes[0],
                )

                reproject(
                    source=labels,
                    destination=aligned,
                    src_transform=src_lbl.transform,
                    src_crs=src_lbl.crs,
                    dst_transform=src_ref.transform,
                    dst_crs=src_ref.crs,
                    resampling=Resampling.nearest,
                )

                if output_path is not None:
                    profile = src_ref.profile.copy()
                    profile.update(
                        dtype=src_lbl.dtypes[0],
                        count=1,
                        compress="lzw",
                    )

                    with rio.open(output_path, "w", **profile) as dst:
                        dst.write(aligned, 1)
                        
        labels = Path("data/labels/sample_2000_2024.tif")
        reference = Path("data/preprocessed/").iterdir().__next__()
        labels_aligned = Path(f"data/labels/{labels.stem}_aligned.tif")
        labels_normalized = Path(f"data/labels/{labels.stem}_normalized.tif")
        
        logger.info(f"Aligning labels to reference {reference.name} ...")
        align_labels_to_reference(
            label_path=labels,
            reference_path=reference,
            output_path=labels_aligned
        )
        
        logger.info(f"Normalizing labels in {labels_aligned.name} ...")
        with rio.open(labels_aligned) as src:
            arr = src.read(1)
            
            year_start_index = self.year_start - 2000
            year_end_index = self.year_end - 2000
            
            arr = np.where((arr >= year_start_index) & (arr <= year_end_index), arr, np.nan)
            arr[np.isnan(arr)] = np.nanmin(arr) - 1
            arr = normalize(arr)
            logger.info(f"Raster labels: {np.unique(arr)}")
            
            profile = src.profile.copy()
            profile.update(
                dtype=rio.float32,
                count=1,
                compress="lzw",
            )
            
            logger.info(f"Saving normalized labels to {labels_normalized.name} ...")
            with rio.open(labels_normalized, "w", **profile) as dst:
                dst.write(arr.astype(rio.float32), 1)
        
        logger.info(f"Checking metadata of normalized labels and reference ...")
        with rio.open(labels_normalized) as lbl:
            logger.info("Label")
            logger.info(f"shape: {lbl.height}, {lbl.width}")
            logger.info(f"bounds: {lbl.bounds}")
            logger.info(f"res: {lbl.res}")
            logger.info(f"crs: {lbl.crs}")

        with rio.open(reference) as s1:
            logger.info("\nSITS")
            logger.info(f"shape: {s1.height}, {s1.width}")
            logger.info(f"bounds: {s1.bounds}")
            logger.info(f"res: {s1.res}")
            logger.info(f"crs: {s1.crs}")