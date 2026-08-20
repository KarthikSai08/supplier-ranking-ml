"""Time-based train/validation/test splitting (shared by the pipeline and the advisor)."""

import pandas as pd

from supplier_ranking.config import TRAIN_END, VAL_END


def split_by_date(
    panel: pd.DataFrame,
    date_col: str = "As_Of_Date",
    train_end: str = TRAIN_END,
    val_end: str = VAL_END,
) -> tuple:
    """Split a panel into (train, val, test) on timestamps strictly below/above the cutoffs."""
    frame = panel.copy()
    frame[date_col] = pd.to_datetime(frame[date_col])
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)

    train = frame[frame[date_col] < train_end_ts].copy()
    val = frame[(frame[date_col] >= train_end_ts) & (frame[date_col] < val_end_ts)].copy()
    test = frame[frame[date_col] >= val_end_ts].copy()
    return train, val, test