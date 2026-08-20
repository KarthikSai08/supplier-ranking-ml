"""Stage 8: TreeSHAP contributions -> plain-language reason for the top pick."""

import numpy as np
import pandas as pd
import xgboost as xgb

from supplier_ranking.models.baseline_regressor import FEATURES

FEATURE_LABELS = {
    "On_Time_Pct": "on-time delivery rate",
    "Rejection_Pct": "rejection rate",
    "Quality_Pct": "quality score",
    "Avg_Delivery_Days": "average delivery days",
    "Avg_Price_Per_Ton": "price per ton",
    "Total_Orders": "order volume",
    "Total_Purchase_Value": "purchase value",
}

LOWER_IS_BETTER = {
    "Rejection_Pct",
    "Avg_Delivery_Days",
    "Avg_Price_Per_Ton",
}

DEFAULT_TOP_K = 3


def shap_contributions(model, features_df: pd.DataFrame) -> np.ndarray:
    """Per-feature contribution matrix (n_rows, n_features + 1); last col = bias."""
    booster = model.get_booster()
    dmatrix = xgb.DMatrix(features_df[FEATURES].astype(float))
    return booster.predict(dmatrix, pred_contribs=True)


def feature_priors(features_df: pd.DataFrame) -> pd.Series:
    """Category-level means per feature, for 'better/worse than average' phrasing."""
    return features_df[FEATURES].mean()


def explain_top_pick(
    model,
    features_row: pd.Series,
    priors: pd.Series,
    supplier_name: str,
    top_k: int = DEFAULT_TOP_K,
) -> str:
    """Build a plain-language 'here's why' sentence for the top-ranked supplier."""
    contribs = shap_contributions(model, features_row.to_frame().T)[0]
    contribution_by_feature = dict(zip(FEATURES, contribs[:-1]))

    ranked_features = sorted(
        contribution_by_feature.items(), key=lambda item: -abs(item[1])
    )[:top_k]

    parts = []
    for feature, _ in ranked_features:
        value = float(features_row[feature])  # type: ignore[arg-type]
        average = float(priors[feature])  # type: ignore[arg-type]
        better = (
            value < average if feature in LOWER_IS_BETTER else value > average
        )
        direction = "better than" if better else "worse than"
        label = FEATURE_LABELS.get(feature, feature)
        parts.append(
            f"{label} of {value:.1f} ({direction} the category average of {average:.1f})"
        )

    return f"{supplier_name} is the top-ranked pick, mainly due to: {'; '.join(parts)}."