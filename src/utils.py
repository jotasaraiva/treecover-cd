from pathlib import Path
import pandas as pd


def load_history(
    run_name: str,
    logs_dir: str | Path = "logs"
) -> pd.DataFrame | None:
    """Per-epoch training history for a run, or None if it has no log.

    A run can be missing a log while still having a checkpoint (an interrupted
    run saves weights on every improvement but writes its log only at the end),
    so callers are expected to handle None.
    """

    history_path = Path(logs_dir) / f"{run_name}.parquet"

    if not history_path.exists():
        return None

    return pd.read_parquet(history_path)
