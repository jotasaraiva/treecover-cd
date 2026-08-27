import torch
from src.models import ConvGRURegressor, ConvLSTMRegressor
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd
from typing import Literal
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

def masked_mse(pred, target, mask, pos_weight=1.0):
    pred = pred.squeeze(1)
    weights = mask * (1.0 + (pos_weight - 1.0) * (target > 0).float())
    se = (pred - target) ** 2 * weights
    return se.sum() / weights.sum().clamp(min=1.0)

def conv_rnn_training(
    input_channels: int,
    kernel_size: int,
    hidden_channels: int,
    head_channels: list[int],
    lrate: float,
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    batch_size: int,
    epochs: int,
    pos_weight: float,
    device: torch.device,
    weights_path: str | Path,
    arch: Literal["ConvGRU", "ConvLSTM"],
    patience: int | None = None,
) -> pd.DataFrame:

    weights_path = Path(weights_path)

    if arch == "ConvGRU":
        model = ConvGRURegressor(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            head_channels=head_channels
        )
    elif arch == "ConvLSTM":
        model = ConvLSTMRegressor(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            head_channels=head_channels
        )
    else:
        raise ValueError("Model unrecognized.")

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lrate)

    history = []
    best_val = float('inf')
    epochs_without_improvement = 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    for e in range(epochs):

        model.train()
        train_dataset.dataset.augment = True
        running_loss = 0.0

        for cube, label, mask in train_loader:
            cube, label, mask = cube.to(device), label.to(device), mask.to(device)

            optimizer.zero_grad()
            pred = model(cube)
            loss = masked_mse(pred, label, mask, pos_weight=pos_weight)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * cube.size(0)

        train_loss = running_loss / len(train_dataset)

        model.eval()
        train_dataset.dataset.augment = False
        running_val_loss = 0.0

        with torch.no_grad():
            for cube, label, mask in val_loader:
                cube, label, mask = cube.to(device), label.to(device), mask.to(device)

                pred = model(cube)
                loss = masked_mse(pred, label, mask)
                running_val_loss += loss.item() * cube.size(0)

        val_loss = running_val_loss / len(val_dataset)

        history.append({
            "epoch": e,
            "train_loss": train_loss,
            "val_loss": val_loss
        })
        logger.info(f"epoch {e+1}/{epochs} | train_loss {train_loss:.5f} | val_loss {val_loss:.5f}")

        if best_val > val_loss:
            best_val = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), weights_path)
        else:
            epochs_without_improvement += 1
            if patience is not None and epochs_without_improvement >= patience:
                logger.info(f"No improvement for {patience} epochs, stopping early at epoch {e+1}/{epochs}.")
                break

    return pd.DataFrame(history)

