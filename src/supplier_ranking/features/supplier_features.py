"""Stages 0-1: supplier eligibility and point-in-time aggregation features."""

import pandas as pd
from typing import Optional, Union

FEATURE_COLUMNS = [
    "Total_Orders",
    "On_Time_Pct",
    "Rejection_Pct",
    "Quality_Pct",
    "Avg_Delivery_Days",
    "Avg_Price_Per_Ton",
    "Total_Purchase_Value",
]


def get_eligible_suppliers(df: pd.DataFrame, product_name: str, as_of_date: str) -> list:
    """Suppliers with history of this product strictly before the as-of date."""
    as_of = pd.Timestamp(as_of_date)
    order_dates = pd.to_datetime(df["Order_Date"])
    mask = (df["Product_Name"] == product_name) & (order_dates < as_of)
    return sorted(df.loc[mask, "Supplier_ID"].unique().tolist())


def get_eligible_suppliers_by_category(df: pd.DataFrame, product_category: str, as_of_date: str) -> list:
    """Suppliers with history in this category strictly before the as-of date."""
    as_of = pd.Timestamp(as_of_date)
    order_dates = pd.to_datetime(df["Order_Date"])
    mask = (df["Product_Category"] == product_category) & (order_dates < as_of)
    return sorted(df.loc[mask, "Supplier_ID"].unique().tolist())


def compute_supplier_features(
    df: pd.DataFrame, supplier_id: str, as_of_date: Union[str, pd.Timestamp], product_name: Optional[str] = None
) -> dict:
    as_of = pd.Timestamp(as_of_date)
    order_dates = pd.to_datetime(df["Order_Date"])
    sub = df[(df["Supplier_ID"] == supplier_id) & (order_dates < as_of)].copy()

    if sub.empty:
        return {col: 0.0 for col in FEATURE_COLUMNS}

    sub["Order_Date"] = pd.to_datetime(sub["Order_Date"])
    sub["Actual_Delivery_Date"] = pd.to_datetime(sub["Actual_Delivery_Date"])
    sub["Lead_Time_Days"] = (sub["Actual_Delivery_Date"] - sub["Order_Date"]).dt.days  # type: ignore[attr-defined]

    total_orders = len(sub)
    on_time_pct = (sub["On_Time_Flag"] == "Yes").mean() * 100
    total_delivered = sub["Delivered_Qty_Tons"].sum()
    total_rejected = sub["Rejected_Qty_Tons"].sum()
    rejection_pct = (total_rejected / total_delivered * 100) if total_delivered > 0 else 0.0
    quality_pct = sub["Quality_Inspection_Score"].mean()
    avg_delivery_days = sub["Lead_Time_Days"].mean()

    price_sub = sub[sub["Product_Name"] == product_name] if product_name else sub
    if price_sub.empty:  # type: ignore[attr-defined]
        price_sub = sub
    avg_price_per_ton = price_sub["Final_Price_Per_Ton"].mean()

    total_purchase_value = (sub["Final_Price_Per_Ton"] * sub["Delivered_Qty_Tons"]).sum()

    return {
        "Total_Orders": total_orders,
        "On_Time_Pct": round(on_time_pct, 2),
        "Rejection_Pct": round(rejection_pct, 2),
        "Quality_Pct": round(quality_pct, 2),
        "Avg_Delivery_Days": round(avg_delivery_days, 2),
        "Avg_Price_Per_Ton": round(avg_price_per_ton, 2),
        "Total_Purchase_Value": round(total_purchase_value, 2),
    }


def compute_all_supplier_features(
    df: pd.DataFrame, as_of_date: Union[str, pd.Timestamp], product_name: Optional[str] = None
) -> pd.DataFrame:
    rows = []
    meta = df[["Supplier_ID", "Supplier_Name", "Product_Category"]].drop_duplicates("Supplier_ID")  # type: ignore[call-overload]
    for supplier_id in df["Supplier_ID"].unique():
        feats = compute_supplier_features(df, supplier_id, as_of_date, product_name=product_name)
        feats["Supplier_ID"] = supplier_id
        rows.append(feats)
    result: pd.DataFrame = pd.DataFrame(rows)
    result = result.merge(meta, on="Supplier_ID", how="left")
    return result[["Supplier_ID", "Supplier_Name", "Product_Category"] + FEATURE_COLUMNS]  # type: ignore[return-value]
