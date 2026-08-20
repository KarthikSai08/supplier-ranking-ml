"""Pipeline orchestration: snapshot panel building plus the stage runners."""

import logging
from typing import Iterable

import pandas as pd

from supplier_ranking.config import (
    DEFAULT_PANEL_OUTPUT,
    PIPELINE_MIN_ORDERS,
    SNAPSHOT_END,
    SNAPSHOT_FREQ,
    SNAPSHOT_START,
    SUBMODEL_MIN_HISTORY,
)
from supplier_ranking.evaluation.metrics import evaluate_ranking
from supplier_ranking.evaluation.splits import split_by_date
from supplier_ranking.features.order_history_features import build_order_level_history_features
from supplier_ranking.features.scoring import compute_composite_score
from supplier_ranking.features.supplier_features import compute_all_supplier_features
from supplier_ranking.io import load_dataset
from supplier_ranking.models import baseline_regressor, ltr_ranker
from supplier_ranking.models.submodels import SUBMODELS, evaluate_submodel, find_best_threshold, train_submodel

logger = logging.getLogger(__name__)


def build_snapshot_panel(
    df: pd.DataFrame, snapshot_dates: Iterable[pd.Timestamp], min_orders: int = PIPELINE_MIN_ORDERS
) -> pd.DataFrame:
    """Stack weekly point-in-time composite scores into a panel keyed by Query_ID."""
    frames = []
    for as_of in snapshot_dates:
        feats = compute_all_supplier_features(df, as_of)
        feats = feats.loc[feats["Total_Orders"] >= min_orders].copy()
        if feats.empty:
            continue
        scored = compute_composite_score(feats)
        scored["As_Of_Date"] = pd.Timestamp(as_of)
        frames.append(scored)
    panel = pd.concat(frames, ignore_index=True)
    panel["Query_ID"] = panel["As_Of_Date"].astype(str) + "_" + panel["Product_Category"]
    return panel


def build_panel(
    data_path,
    snapshot_start: str = SNAPSHOT_START,
    snapshot_end: str = SNAPSHOT_END,
) -> pd.DataFrame:
    """Build the snapshot panel and write it to DEFAULT_PANEL_OUTPUT."""
    df = load_dataset(data_path)
    snapshot_dates = pd.date_range(snapshot_start, snapshot_end, freq=SNAPSHOT_FREQ)
    panel = build_snapshot_panel(df, snapshot_dates)
    logger.info("Panel built: %d rows across %d query groups", len(panel), panel["Query_ID"].nunique())
    panel.to_csv(DEFAULT_PANEL_OUTPUT, index=False)
    logger.info("Panel written to %s", DEFAULT_PANEL_OUTPUT)
    return panel


def _warn_on_empty_splits(splits: tuple) -> None:
    """Warn when a time split is empty (usually a narrowed snapshot range)."""
    for name, split in zip(("train", "val", "test"), splits):
        if split.empty:
            logger.warning("%s split is empty - widen the snapshot range in config.py for full evaluation", name)


def run_baseline(panel: pd.DataFrame) -> None:
    """Train/evaluate the baseline regressor, then compare the LTR ranker against it."""
    train, val, test = split_by_date(panel)
    _warn_on_empty_splits((train, val, test))
    if train.empty or val.empty:
        return

    model = baseline_regressor.train_baseline_regressor(train, val)
    val_metrics, _ = baseline_regressor.evaluate_baseline_regressor(model, val)
    logger.info("Baseline val: %s", {k: round(v, 3) for k, v in val_metrics.items()})
    if test.empty:
        return
    test_metrics, test_preds = baseline_regressor.evaluate_baseline_regressor(model, test)
    logger.info("Baseline test: %s", {k: round(v, 3) for k, v in test_metrics.items()})
    run_ltr(panel, model, test, test_preds)


def run_ltr(panel: pd.DataFrame, baseline_model, test: pd.DataFrame, baseline_preds) -> None:
    """Train the LambdaMART ranker and compare ranking quality vs baseline-as-ranker."""
    train, val, _ = split_by_date(panel)
    grades_train = ltr_ranker.make_relevance_grades(train["Composite_Score"], train["Composite_Score"])
    grades_test = ltr_ranker.make_relevance_grades(train["Composite_Score"], test["Composite_Score"])
    train_r = train.assign(Relevance=grades_train)
    val_r = val.assign(Relevance=ltr_ranker.make_relevance_grades(train["Composite_Score"], val["Composite_Score"]))
    test_r = test.assign(Relevance=grades_test)

    model = ltr_ranker.train_ltr_ranker(train_r, val_r)
    test_sorted = test_r.sort_values("Query_ID").assign(
        LTR_Score=ltr_ranker.score_ranker(model, test_r.sort_values("Query_ID"))
    )
    ltr_groups = [
        (g["Relevance"].values, g["LTR_Score"].values)
        for _, g in test_sorted.groupby("Query_ID")
        if len(g) >= 2
    ]

    baseline_df = test_r.assign(Baseline_Score=baseline_preds).sort_values("Query_ID")
    baseline_groups = [
        (g["Relevance"].values, g["Baseline_Score"].values)
        for _, g in baseline_df.groupby("Query_ID")
        if len(g) >= 2
    ]

    logger.info("LTR metrics: %s", {k: round(v, 3) for k, v in evaluate_ranking(ltr_groups).items()})
    logger.info("Baseline-as-ranker metrics: %s", {k: round(v, 3) for k, v in evaluate_ranking(baseline_groups).items()})


def run_submodels(data_path) -> None:
    """Train/evaluate the Stage 5 sub-models from the SUBMODELS registry."""
    df = load_dataset(data_path)
    hist = build_order_level_history_features(df, min_history=SUBMODEL_MIN_HISTORY)
    train, val, test = split_by_date(hist, date_col="Order_Date")
    _warn_on_empty_splits((train, val, test))
    logger.info("Sub-model split: train=%d val=%d test=%d", len(train), len(val), len(test))
    if train.empty or val.empty:
        return

    for name, spec in SUBMODELS.items():
        model = train_submodel(spec, train, val)
        logger.info("%s trained on train=%d", name, len(train))
        if test.empty:
            continue
        metrics, _ = evaluate_submodel(spec, model, test)
        logger.info("%s metrics: %s", name, {k: round(v, 3) for k, v in metrics.items()})
        if spec.kind == "classification":
            threshold = find_best_threshold(model, val)
            metrics, _ = evaluate_submodel(spec, model, test, threshold=threshold)
            logger.info("Risk metrics (threshold=%s): %s", threshold, {k: round(v, 3) for k, v in metrics.items()})