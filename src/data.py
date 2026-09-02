from rasterio.warp import reproject, Resampling
from pathlib import Path
import rasterio as rio
import numpy as np
import kagglehub
import shutil
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

KAGGLE_DATASET = "jotasaraiva/treecover-cd-data"

class DataPipeline:

    def __init__(
        self,
        label_path: str | Path,
        raw_path: str | Path,
        preproc_path: str | Path,
        stats_path: str | Path = "data/aggregated/norm_stats.json",
        year_start: int = 2018,
        year_end: int = 2024,
    ) -> None:
        self.year_start = year_start
        self.year_end = year_end
        self.label_path = label_path
        self.raw_path = raw_path
        self.preproc_path = preproc_path
        self.stats_path = stats_path

    def raw(self):

        raw_dir = Path(self.raw_path)
        raw_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Baixando dataset {KAGGLE_DATASET} do Kaggle ...")
        dataset_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET))

        source_dir = dataset_dir / raw_dir.name
        if not source_dir.exists():
            raise FileNotFoundError(
                f"Pasta '{raw_dir.name}' não encontrada em {dataset_dir}. "
                f"Conteúdo disponível: {[p.name for p in dataset_dir.iterdir()]}"
            )

        files = list(source_dir.glob("*.tif"))
        for f in files:
            shutil.copy2(f, raw_dir / f.name)

        logger.info(f"{len(files)} arquivos copiados de {source_dir} para {raw_dir}")

    def preprocessed(self):

        raw_dir = Path(self.raw_path)
        out_dir = Path(self.preproc_path)

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

            logger.info(f"Pré-processando {f.stem} ...")
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

        logger.info("Computando estatísticas de normalização por mês ...")
        stats_path = Path(self.stats_path)
        stats_path.parent.mkdir(parents=True, exist_ok=True)

        months = sorted({
            f.stem.split(".")[0]
            for f in out_dir.glob("*.tif")
        })

        norm_stats = {}
        for month in months:
            vv_path = out_dir / f"{month}.VV.tif"
            vh_path = out_dir / f"{month}.VH.tif"

            with rio.open(vv_path) as src:
                vv = src.read(1).astype(np.float32)

            with rio.open(vh_path) as src:
                vh = src.read(1).astype(np.float32)

            norm_stats[month] = {
                "vv": [float(np.nanmean(vv)), float(np.nanstd(vv) + 1e-6)],
                "vh": [float(np.nanmean(vh)), float(np.nanstd(vh) + 1e-6)]
            }

        with open(stats_path, "w") as f:
            json.dump(norm_stats, f, indent=2)

        logger.info(f"Dados de normalização salvos em {stats_path}")

    def labels(self, outfile: str | Path):
        
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
                        
        labels = Path(self.label_path)
        reference = Path(self.preproc_path).iterdir().__next__()
        labels_aligned = Path(f"data/labels/{labels.stem}_aligned.tif")
        labels_normalized = Path(outfile)
        
        logger.info(f"Alinhando rótulos para a referência {reference.name} ...")
        align_labels_to_reference(
            label_path=labels,
            reference_path=reference,
            output_path=labels_aligned
        )
        
        logger.info(f"Normalizando rótulos em {labels_aligned.name} ...")
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
            
            logger.info(f"Salvando rótulos normalizados em {labels_normalized.name} ...")
            with rio.open(labels_normalized, "w", **profile) as dst:
                dst.write(arr.astype(rio.float32), 1)
        
        logger.info(f"Verificando metadados dos rótulos normalizados e da referência ...")
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