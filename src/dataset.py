from pathlib import Path
from rasterio.windows import Window
import rasterio as rio
import torch
from torch.utils.data import Dataset
import numpy as np
import json

class SARDataset(Dataset):

    def __init__(
        self,
        data_dir: str | Path,
        label_path: str | Path,
        stats_path: str | Path = "data/aggregated/norm_stats.json",
        patch_size: int = 256,
        augment: bool = False
    ):

        self.data_dir = Path(data_dir)
        self.label_path = Path(label_path)
        self.patch_size = patch_size
        self.augment = augment
        self.patches = []

        with open(stats_path) as f:
            self.norm_stats = json.load(f)

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

        valid = np.isfinite(patch)

        padded = np.zeros(
            (self.patch_size, self.patch_size),
            dtype=np.float32
        )

        padded[
            :read_height,
            :read_width
        ] = np.where(valid, patch, 0.0)

        mask = np.zeros(
            (self.patch_size, self.patch_size),
            dtype=np.float32
        )

        mask[
            :read_height,
            :read_width
        ] = valid.astype(np.float32)

        return padded, mask

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

            vv, patch_mask = self._read_patch(vv_path, row, col)
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

        if self.augment:
            if torch.rand(1).item() < 0.5:
                cube = torch.flip(cube, dims=[3])
                label = torch.flip(label, dims=[1])
                mask = torch.flip(mask, dims=[1])

            if torch.rand(1).item() < 0.5:
                cube = torch.flip(cube, dims=[2])
                label = torch.flip(label, dims=[0])
                mask = torch.flip(mask, dims=[0])

        return cube, label, mask
