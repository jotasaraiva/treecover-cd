from pathlib import Path
import rasterio as rio
import torch
from torch.utils.data import Dataset
import numpy as np

class SARDataset(Dataset):

    def __init__(self, data_dir, label_path):

        self.data_dir = Path(data_dir)
        self.label_path = Path(label_path)

        self.months = sorted({
            f.stem.split(".")[0]
            for f in self.data_dir.glob("*.tif")
        })

    def __len__(self):
        return 1

    def __getitem__(self, idx):

        images = []

        for month in self.months:

            with rio.open(self.data_dir / f"{month}.VV.tif") as src:
                vv = src.read(1)

            with rio.open(self.data_dir / f"{month}.VH.tif") as src:
                vh = src.read(1)

            images.append(np.stack([vv, vh]))

        cube = np.stack(images)

        with rio.open(self.label_path) as src:
            label = src.read(1)

        cube = torch.from_numpy(cube).float()
        label = torch.from_numpy(label).float()

        return cube, label
            
            