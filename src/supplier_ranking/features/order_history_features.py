"""Stage 5 inputs: order-level rolling history features and per-order targets."""

import pandas as pd


ROLL_WINDOW = 20
ROLL_MIN_PERIODS = 5
PRICE_ROLL_MIN_PERIODS = 3


def build_order_level_history_features(df: pd.DataFrame, min_history: int = 10) -> pd.DataFrame:
    d = df.copy()
    d["Order_Date"] = pd.to_datetime(d["Order_Date"])
    d["Actual_Delivery_Date"] = pd.to_datetime(d["Actual_Delivery_Date"])
    d["Lead_Time_Days"] = (d["Actual_Delivery_Date"] - d["Order_Date"]).dt.days
    d["Is_On_Time"] = (d["On_Time_Flag"] == "Yes").astype(int)
    d["Category_Code"] = d["Product_Category"].astype("category").cat.codes
    d = d.sort_values(["Supplier_ID", "Order_Date"]).reset_index(drop=True)

    by_supplier = d.groupby("Supplier_ID")
    by_supplier_product = d.groupby(["Supplier_ID", "Product_Name"])

    d["Hist_Order_Count"] = by_supplier.cumcount()

    def rolling_mean(s, min_periods):
        return s.shift(1).rolling(ROLL_WINDOW, min_periods=min_periods).mean()

    def rolling_sum(s):
        return s.shift(1).rolling(ROLL_WINDOW, min_periods=ROLL_MIN_PERIODS).sum()

    d["Hist_On_Time_Pct"] = by_supplier["Is_On_Time"].transform(lambda s: rolling_mean(s, ROLL_MIN_PERIODS)) * 100
    d["Hist_Quality_Pct"] = by_supplier["Quality_Inspection_Score"].transform(lambda s: rolling_mean(s, ROLL_MIN_PERIODS))
    d["Hist_Avg_Delivery_Days"] = by_supplier["Lead_Time_Days"].transform(lambda s: rolling_mean(s, ROLL_MIN_PERIODS))
    d["Hist_Avg_Price_Per_Ton"] = by_supplier_product["Final_Price_Per_Ton"].transform(
        lambda s: rolling_mean(s, PRICE_ROLL_MIN_PERIODS)
    )

    d["Roll_Delivered"] = by_supplier["Delivered_Qty_Tons"].transform(rolling_sum)
    d["Roll_Rejected"] = by_supplier["Rejected_Qty_Tons"].transform(rolling_sum)
    d["Hist_Rejection_Pct"] = (d["Roll_Rejected"] / d["Roll_Delivered"] * 100).fillna(0.0)

    d["Target_Price"] = d["Final_Price_Per_Ton"]
    d["Target_Delivery_Days"] = d["Lead_Time_Days"]
    d["Target_Quality"] = d["Quality_Inspection_Score"]
    d["Target_Risk"] = ((d["Goods_Returned_Flag"] == "Yes") | (d["Rejection_Pct_Order"] > 5.0)).astype(int)

    feature_cols = [
        "Hist_Order_Count",
        "Hist_On_Time_Pct",
        "Hist_Rejection_Pct",
        "Hist_Quality_Pct",
        "Hist_Avg_Delivery_Days",
        "Hist_Avg_Price_Per_Ton",
        "Category_Code",
    ]
    target_cols = ["Target_Price", "Target_Delivery_Days", "Target_Quality", "Target_Risk"]
    keep_cols = ["Supplier_ID", "Supplier_Name", "Product_Category", "Product_Name", "Order_Date"] + feature_cols + target_cols

    result = d[keep_cols].copy()
    result = result[result["Hist_Order_Count"] >= min_history].dropna(subset=feature_cols).reset_index(drop=True)  # type: ignore[index,arg-type]
    return result


ORDER_HISTORY_FEATURES = [
    "Hist_Order_Count",
    "Hist_On_Time_Pct",
    "Hist_Rejection_Pct",
    "Hist_Quality_Pct",
    "Hist_Avg_Delivery_Days",
    "Hist_Avg_Price_Per_Ton",
    "Category_Code",
]
