"""Stage 6: cold-start handling - shrink low-history features toward the category mean."""

import pandas as pd

SHRINKABLE_FEATURES = [
    "On_Time_Pct",
    "Rejection_Pct",
    "Quality_Pct",
    "Avg_Delivery_Days",
    "Avg_Price_Per_Ton",
]

DEFAULT_K = 5


def category_priors(features_df: pd.DataFrame) -> pd.DataFrame:
    """Per-category mean of each shrinkable feature, used as the prior."""
    priors: pd.DataFrame = features_df.groupby("Product_Category")[SHRINKABLE_FEATURES].mean()  # type: ignore[assignment]
    return priors


def shrink_features(features_df: pd.DataFrame, k: float = DEFAULT_K) -> pd.DataFrame:
    """Return a copy with shrinkable features pulled toward the category mean.

    Suppliers with little history (small Total_Orders) end up close to the
    category baseline; suppliers with deep history keep their own numbers.
    """
    df = features_df.copy()
    priors = category_priors(df)
    n = df["Total_Orders"].clip(lower=1)
    for feature in SHRINKABLE_FEATURES:
        prior = df["Product_Category"].map(priors[feature]).fillna(0.0)  # type: ignore[arg-type]
        df[feature] = (n * df[feature] + k * prior) / (n + k)
    return df