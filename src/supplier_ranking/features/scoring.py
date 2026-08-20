"""Stage 2: weighted 0-100 composite score, min-max normalized per product category."""

import pandas as pd

DEFAULT_WEIGHTS = {
    "On_Time_Pct": 0.22,
    "Rejection_Pct": 0.22,
    "Quality_Pct": 0.22,
    "Avg_Delivery_Days": 0.17,
    "Avg_Price_Per_Ton": 0.17,
}

HIGHER_IS_BETTER = {
    "On_Time_Pct": True,
    "Rejection_Pct": False,
    "Quality_Pct": True,
    "Avg_Delivery_Days": False,
    "Avg_Price_Per_Ton": False,
}


def normalize_column(series: pd.Series, higher_is_better: bool) -> pd.Series:
    col_min = series.min()
    col_max = series.max()
    if col_max == col_min:
        return pd.Series(0.5, index=series.index)
    norm = (series - col_min) / (col_max - col_min)
    return norm if higher_is_better else 1 - norm


def compute_composite_score(features_df: pd.DataFrame, category_col: str = "Product_Category", weights: dict = None) -> pd.DataFrame:  # type: ignore[assignment]
    weights = weights or DEFAULT_WEIGHTS
    result = features_df.copy()
    result["Composite_Score"] = 0.0

    for _, group in result.groupby(category_col):
        idx = group.index
        score = pd.Series(0.0, index=idx)
        for feature, weight in weights.items():
            norm = normalize_column(group[feature], HIGHER_IS_BETTER[feature])  # type: ignore[arg-type]
            score = score + norm * weight
        result.loc[idx, "Composite_Score"] = (score * 100).round(2)

    return result
