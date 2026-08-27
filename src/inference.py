from pathlib import Path
from torch.utils.data import DataLoader
import rasterio as rio
import numpy as np
import torch
import pandas as pd
import logging

from src.models import ConvGRURegressor, ConvLSTMRegressor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

# Gate conv that is always present in each architecture's cell, used to
# identify the architecture and read its dimensions off the state dict.
ARCH_PROBES = {
    "ConvGRU": ("encoder.cell.conv_z.weight", ConvGRURegressor),
    "ConvLSTM": ("encoder.cell.conv_i.weight", ConvLSTMRegressor),
}

def infer_hyperparams(state_dict: dict) -> dict:
    """Recover the constructor kwargs of a saved regressor from its state dict.

    Checkpoints store weights only, so the architecture has to be rebuilt before
    the weights can be loaded. Everything needed is recoverable from the tensor
    shapes, which keeps this independent of checkpoint file naming.
    """

    arch = None
    for name, (probe, _) in ARCH_PROBES.items():
        if probe in state_dict:
            arch = name
            break

    if arch is None:
        raise ValueError(
            "Could not identify architecture from state dict. "
            f"Expected one of {[p for p, _ in ARCH_PROBES.values()]}."
        )

    # Gate convs are Conv2d(input + hidden -> hidden, kernel_size)
    gate = state_dict[ARCH_PROBES[arch][0]]
    hidden_channels = gate.shape[0]
    kernel_size = gate.shape[2]
    input_channels = gate.shape[1] - hidden_channels

    # The head is Conv2d/activation pairs then a final 1x1 conv to one channel,
    # so every head conv but the last contributes one entry to head_channels.
    head_indices = sorted(
        int(k.split(".")[1])
        for k in state_dict
        if k.startswith("head.") and k.endswith(".weight")
    )
    head_channels = [
        state_dict[f"head.{i}.weight"].shape[0]
        for i in head_indices[:-1]
    ]

    return {
        "arch": arch,
        "input_channels": input_channels,
        "hidden_channels": hidden_channels,
        "kernel_size": kernel_size,
        "head_channels": head_channels,
    }

def load_checkpoint(
    weights_path: str | Path,
    device: torch.device | None = None
) -> tuple[torch.nn.Module, dict]:
    """Rebuild the architecture a checkpoint was trained with and load its weights.

    Returns the model in eval mode alongside the recovered hyperparameters.
    """

    weights_path = Path(weights_path)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    hyperparams = infer_hyperparams(state_dict)

    _, regressor = ARCH_PROBES[hyperparams["arch"]]
    model = regressor(**{k: v for k, v in hyperparams.items() if k != "arch"})

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, hyperparams

def predict_full_extent(
    model: torch.nn.Module,
    dataset,
    device: torch.device | None = None,
    batch_size: int = 2
) -> np.ndarray:
    """Run the model over every patch and stitch the results into one raster.

    Patches are predicted independently and written into place without blending,
    so tile seams can be visible where neighbouring patches disagree.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_pred = np.zeros((dataset.height, dataset.width), dtype=np.float32)

    # Augmentation would flip patches out of alignment with their position
    was_augmenting = dataset.augment
    dataset.augment = False

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model.eval()

    try:
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
    finally:
        dataset.augment = was_augmenting

    return full_pred

def reference_extent(dataset) -> tuple[np.ndarray, np.ndarray]:
    """Read the full-extent label and validity mask the predictions align to.

    Reads the rasters directly rather than iterating the dataset, which would
    pull all T months of both bands per patch just to recover two arrays.
    """

    with rio.open(dataset.label_path) as src:
        label = src.read(1).astype(np.float32)

    # SARDataset takes its mask from the first month's VV band
    first_month = dataset.months[0]
    with rio.open(dataset.data_dir / f"{first_month}.VV.tif") as src:
        reference = src.read(1)

    mask = np.isfinite(reference).astype(np.float32)
    label = np.where(np.isfinite(label), label, 0.0)

    return label, mask

def masked_mae(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None = None
) -> float:
    """Mean absolute error over valid pixels.

    In the same 0-1 units as the recency target, so the result reads directly as
    "average distance from the true recency value".
    """

    if mask is None:
        valid = np.ones(pred.shape, dtype=bool)
    else:
        valid = mask > 0

    if not valid.any():
        return float("nan")

    return float(np.abs(pred[valid] - target[valid]).mean())

def list_checkpoints(weights_dir: str | Path = "checkpoints") -> list[Path]:
    """Every checkpoint in the directory, sorted by name."""
    return sorted(Path(weights_dir).glob("*.pth"))

def best_val_loss(
    run_name: str,
    logs_dir: str | Path = "logs"
) -> float | None:
    """Best validation loss recorded for a run, or None if it has no log."""

    history_path = Path(logs_dir) / f"{run_name}.parquet"

    if not history_path.exists():
        return None

    return float(pd.read_parquet(history_path).val_loss.min())
