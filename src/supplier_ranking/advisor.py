"""Advisor pipeline: eligibility -> cold start -> ML score -> blend -> explain."""

from datetime import datetime
from functools import lru_cache

import numpy as np
import pandas as pd

from supplier_ranking.cold_start import shrink_features
from supplier_ranking.config import DEFAULT_ALPHA, PIPELINE_MIN_ORDERS, SNAPSHOT_END, SNAPSHOT_START
from supplier_ranking.evaluation.splits import split_by_date
from supplier_ranking.explain import explain_top_pick, feature_priors
from supplier_ranking.features.supplier_features import compute_all_supplier_features, get_eligible_suppliers
from supplier_ranking.io import load_dataset
from supplier_ranking.models.baseline_regressor import FEATURES as BASELINE_FEATURES
from supplier_ranking.models.baseline_regressor import train_baseline_regressor
from supplier_ranking.pipeline.runner import build_snapshot_panel
from supplier_ranking.weight_blend import blend_scores, user_weighted_score

DEFAULT_SNAPSHOT_RANGE = (SNAPSHOT_START, SNAPSHOT_END)


@lru_cache(maxsize=8)
def _trained_baseline(data_path: str, snapshot_range: tuple):
    """Train and cache the baseline regressor per dataset."""
    df = load_dataset(data_path)
    snapshot_dates = pd.date_range(snapshot_range[0], snapshot_range[1], freq="W-MON")
    panel = build_snapshot_panel(df, snapshot_dates, min_orders=PIPELINE_MIN_ORDERS)
    panel["As_Of_Date"] = pd.to_datetime(panel["As_Of_Date"])
    train, val, _ = split_by_date(panel)
    return train_baseline_regressor(train, val)


def advise(
    data_path,
    product_name,
    as_of_date=None,
    quantity=None,
    weights=None,
    alpha=DEFAULT_ALPHA,
    snapshot_range=DEFAULT_SNAPSHOT_RANGE,
):
    """Recommend a supplier for a new PO; returns {top_pick, reason, ranked, as_of_date, eligible_count}."""
    df = load_dataset(data_path)
    as_of = as_of_date or datetime.now().strftime("%Y-%m-%d")

    eligible = get_eligible_suppliers(df, product_name, as_of)
    if not eligible:
        return {
            "top_pick": None,
            "reason": "No supplier has supplied this product before the as-of date.",
            "ranked": pd.DataFrame(),
            "as_of_date": as_of,
            "eligible_count": 0,
        }

    features = compute_all_supplier_features(df, as_of, product_name=product_name)
    features = features[features["Supplier_ID"].isin(eligible)].copy()
    shrunk = shrink_features(features)

    model = _trained_baseline(str(data_path), tuple(snapshot_range))
    ml_scores = model.predict(shrunk[BASELINE_FEATURES].astype(float))
    user_scores = user_weighted_score(shrunk, weights)
    final_scores = blend_scores(ml_scores, user_scores, alpha)

    ranked = shrunk.copy()
    ranked["ML_Score"] = np.round(ml_scores, 2)
    ranked["User_Score"] = user_scores
    ranked["Final_Score"] = final_scores
    ranked = ranked.sort_values("Final_Score", ascending=False).reset_index(drop=True)
    ranked.insert(0, "Rank", range(1, len(ranked) + 1))

    top_row = ranked.iloc[0]
    priors = feature_priors(features)
    reason = explain_top_pick(model, top_row[BASELINE_FEATURES], priors, top_row["Supplier_Name"])

    return {
        "top_pick": top_row,
        "reason": reason,
        "ranked": ranked,
        "as_of_date": as_of,
        "eligible_count": len(ranked),
    }