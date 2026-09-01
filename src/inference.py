from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import logging
from skimage.filters import threshold_otsu
from scipy.ndimage import median_filter

from src.dataset import SARDataset
from src.models import ConvGRURegressor, ConvLSTMRegressor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


def masked_mae(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None = None
) -> float:

    if mask is None:
        valid = np.ones(pred.shape, dtype=bool)
    else:
        valid = mask > 0

    if not valid.any():
        return float("nan")

    return float(np.abs(pred[valid] - target[valid]).mean())


def masked_rmse(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None = None
) -> float:

    if mask is None:
        valid = np.ones(pred.shape, dtype=bool)
    else:
        valid = mask > 0

    if not valid.any():
        return float("nan")

    return float(np.sqrt(((pred[valid] - target[valid]) ** 2).mean()))


def load_weights(
    arch: ConvLSTMRegressor | ConvGRURegressor,
    weights_path: str | Path,
    device: torch.device | None = None,
    **kwargs
) -> nn.Module:
    weights_path = Path(weights_path)
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model = arch(**kwargs)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def conv_rnn_predict(
    model: nn.Module,
    dataset: SARDataset,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    full_pred = np.zeros((dataset.height, dataset.width), dtype=np.float32)
    dataset.augment = False
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    idx = 0
    with torch.no_grad():
        for cube, _, _ in loader:
            preds = model(cube.to(device)).squeeze(1).cpu().numpy()
            
            for pred in preds:
                row, col = dataset.patches[idx]
                read_height = min(dataset.patch_size, dataset.height - row)
                read_width = min(dataset.patch_size, dataset.width - col)
                 
                full_pred[
                    row:row + read_height,
                    col:col + read_width
                ] = pred[:read_height, :read_width]
                
                idx += 1
                
    return postprocess(full_pred)


def postprocess(arr: np.ndarray, size: int = 5) -> np.ndarray:
    assert len(arr.shape) == 2, "Input array must be 2-dimensional"
    
    th = threshold_otsu(arr)
    arr = np.where(arr <= th, 0, arr).astype(np.float32)
    arr = median_filter(arr, size=size)
    return arr