from pathlib import Path
import rasterio as rio
from rasterio.windows import Window
import torch
from torch.utils.data import Dataset
import numpy as np

class SARDataset(Dataset):

    def __init__(self, data_dir: str | Path, label_path: str | Path, patch_size: int = 256):

        self.data_dir = Path(data_dir)
        self.label_path = Path(label_path)
        self.patch_size = patch_size
        self.patches = []
        self.norm_stats = {}
        
        self.months = sorted({
            f.stem.split(".")[0]
            for f in self.data_dir.glob("*.tif")
        })
        
        with rio.open(self.label_path) as src:
            self.height = src.height
            self.width = src.width
            
        for row in range(0, self.height, self.patch_size):
            for col in range(0, self.width, self.patch_size):
                self.patches.append((row, col))
                
        for month in self.months:
            vv_path = self.data_dir / f"{month}.VV.tif"
            vh_path = self.data_dir / f"{month}.VH.tif"
            
            vv_mean, vv_std = self._compute_stats(vv_path)
            vh_mean, vh_std = self._compute_stats(vh_path)
            
            self.norm_stats[month] = {
                "vv": (vv_mean, vv_std),
                "vh": (vh_mean, vh_std)
            }

    def _read_patch(self, path, row, col):

        with rio.open(path) as src:

            read_height = min(
                self.patch_size,
                self.height - row
            )

            read_width = min(
                self.patch_size,
                self.width - col
            )

            window = Window(
                col,
                row,
                read_width,
                read_height
            )

            patch = src.read(
                1,
                window=window
            )

        padded = np.zeros(
            (self.patch_size, self.patch_size),
            dtype=np.float32
        )

        padded[
            :read_height,
            :read_width
        ] = patch
        
        mask = np.zeros(
            (self.patch_size, self.patch_size),
            dtype=np.float32
        )

        mask[
            :read_height,
            :read_width
        ] = 1.0

        return padded, mask

    def _compute_stats(self, path):

        with rio.open(path) as src:
            x = src.read(1).astype(np.float32)

        mean = x.mean()
        std = x.std() + 1e-6

        return mean, std

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        
        row, col = self.patches[idx]
        
        T = len(self.months)

        cube = np.zeros(
            (
                T,
                2,
                self.patch_size,
                self.patch_size
            ),
            dtype=np.float32
        )
        
        for t, month in enumerate(self.months):

            vv_path = self.data_dir / f"{month}.VV.tif"
            vh_path = self.data_dir / f"{month}.VH.tif"

            vv, patch_mask  = self._read_patch(vv_path, row, col)
            vh, _ = self._read_patch(vh_path, row, col)
            
            vv_mean, vv_std = self.norm_stats[month]["vv"]
            vh_mean, vh_std = self.norm_stats[month]["vh"]
            
            vv = (vv - vv_mean) / vv_std
            vh = (vh - vh_mean) / vh_std

            cube[t, 0] = vv
            cube[t, 1] = vh
            
            if t == 0:
                mask = patch_mask
        
        label, _ = self._read_patch(self.label_path, row, col)
        
        cube = torch.from_numpy(cube).float()
        label = torch.from_numpy(label).float()
        mask = torch.from_numpy(mask).float()

        return cube, label, mask
            
            