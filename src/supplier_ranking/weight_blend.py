"""Stage 7: blend the ML score with the user's priority weights (alpha)."""

import pandas as pd

DIMENSIONS = {
    "price": ("Avg_Price_Per_Ton", False),
    "on_time": ("On_Time_Pct", True),
    "quality": ("Quality_Pct", True),
    "delivery": ("Avg_Delivery_Days", False),
    "rejection": ("Rejection_Pct", False),
}

DEFAULT_ALPHA = 0.5


def user_weighted_score(features_df: pd.DataFrame, weights: dict = None) -> pd.Series:  # type: ignore[assignment]
    """0-100 score from user priorities via per-dimension min-max normalization.

    weights maps dimension names (price, on_time, quality, delivery, rejection)
    to importance values (any positive scale, e.g. 0-10 or 0-1); they are
    normalized internally, so only relative proportions matter.
    """
    weights = weights or {name: 1.0 for name in DIMENSIONS}
    score = pd.Series(0.0, index=features_df.index, dtype=float)
    total_weight = 0.0
    for name, (column, higher_is_better) in DIMENSIONS.items():
        weight = float(weights.get(name, 0.0))
        if weight <= 0:
            continue
        total_weight += weight
        series: pd.Series = features_df[column].astype(float)  # type: ignore[assignment]
        col_min, col_max = float(series.min()), float(series.max())
        if col_max == col_min:
            norm = pd.Series(0.5, index=features_df.index)
        else:
            norm = (series - col_min) / (col_max - col_min)
            if not higher_is_better:
                norm = 1 - norm
        score = score + norm * weight
    if total_weight > 0:
        return (score / total_weight * 100).round(2)
    return pd.Series(0.0, index=features_df.index)


def blend_scores(ml_scores, user_scores, alpha: float = DEFAULT_ALPHA) -> pd.Series:
    """Blend model scores with user-weighted scores.

    alpha = weight on the model score (0 = only user priorities,
    1 = only the model); final = alpha * ML + (1 - alpha) * user.
    """
    return (alpha * ml_scores + (1 - alpha) * user_scores).round(2)