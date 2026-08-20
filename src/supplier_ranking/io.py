"""Data access layer: the only place raw order-history CSVs are read."""

from pathlib import Path
from typing import Union

import pandas as pd

REQUIRED_COLUMNS = [
    "Order_ID",
    "Supplier_ID",
    "Supplier_Name",
    "Product_Category",
    "Product_Name",
    "Order_Date",
    "Actual_Delivery_Date",
    "On_Time_Flag",
    "Delivered_Qty_Tons",
    "Rejected_Qty_Tons",
    "Quality_Inspection_Score",
    "Final_Price_Per_Ton",
]


def load_dataset(path: Union[str, Path]) -> pd.DataFrame:
    """Read the order-history CSV and fail fast with a clear message if required columns are missing."""
    df = pd.read_csv(path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    return df