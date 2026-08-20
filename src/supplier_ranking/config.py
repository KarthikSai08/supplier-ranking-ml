"""Central configuration: the single source of truth for shared constants."""

from pathlib import Path

# Time-based train/validation/test split (anti-leakage)
TRAIN_END = "2024-01-01"
VAL_END = "2026-01-01"

# Minimum order history (panel vs interactive flows)
PIPELINE_MIN_ORDERS = 10
ADVISOR_MIN_ORDERS = 1
SUBMODEL_MIN_HISTORY = 10

# Default data locations (derived from this file so they work anywhere)
DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parents[2] / "Data" / "raw" / "supplier_ranking_dataset.csv"
)
DEFAULT_PANEL_OUTPUT = (
    Path(__file__).resolve().parents[2] / "Data" / "processed" / "snapshot_panel.csv"
)

# Snapshot schedule: weekly Monday snapshots between these dates
SNAPSHOT_START = "2022-06-01"
SNAPSHOT_END = "2026-07-01"
SNAPSHOT_FREQ = "W-MON"

# Model/user blend: final = alpha * ML + (1 - alpha) * user
DEFAULT_ALPHA = 0.5