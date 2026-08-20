import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from supplier_ranking.pipeline.runner import build_snapshot_panel
from supplier_ranking.features.order_history_features import build_order_level_history_features
from supplier_ranking.models.baseline_regressor import train_baseline_regressor, evaluate_baseline_regressor, FEATURES as BASELINE_FEATURES
from supplier_ranking.models.ltr_ranker import train_ltr_ranker, score_ranker, make_relevance_grades
from supplier_ranking.models.submodels import SUBMODELS, train_submodel, evaluate_submodel, find_best_threshold
from supplier_ranking.evaluation.metrics import evaluate_ranking

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "Data", "raw", "supplier_ranking_dataset.csv")
TRAIN_END = pd.Timestamp("2024-01-01")
VAL_END = pd.Timestamp("2026-01-01")

df = pd.read_csv(DATA_PATH)


def split_by_date(panel, date_col):
    train = panel[panel[date_col] < TRAIN_END].copy()
    val = panel[(panel[date_col] >= TRAIN_END) & (panel[date_col] < VAL_END)].copy()
    test = panel[panel[date_col] >= VAL_END].copy()
    return train, val, test


def stage3_baseline(panel):
    print("=== STAGE 3: Baseline pointwise regressor ===")
    train, val, test = split_by_date(panel, "As_Of_Date")
    print(f"  train={len(train)} val={len(val)} test={len(test)}")

    model = train_baseline_regressor(train, val)
    val_metrics, _ = evaluate_baseline_regressor(model, val)
    test_metrics, test_preds = evaluate_baseline_regressor(model, test)

    print(f"  Val:  MAE={val_metrics['MAE']:.2f} R2={val_metrics['R2']:.3f} Spearman={val_metrics['Spearman']:.3f}")
    print(f"  Test: MAE={test_metrics['MAE']:.2f} R2={test_metrics['R2']:.3f} Spearman={test_metrics['Spearman']:.3f}")

    importances = dict(zip(BASELINE_FEATURES, model.feature_importances_))
    ranked_importance = sorted(importances.items(), key=lambda x: -x[1])
    print("  Feature importance (top 4):")
    for feat, imp in ranked_importance[:4]:
        print(f"    {feat}: {imp:.3f}")

    passed = test_metrics["R2"] > 0.5 and test_metrics["Spearman"] > 0.5
    print(f"Stage 3 result: {'PASS' if passed else 'FAIL'}\n")
    return model, test, test_preds, passed


def stage4_ltr(panel, baseline_model, baseline_test_preds, baseline_test_df):
    print("=== STAGE 4: Learning-to-Rank (LambdaMART) ===")
    train, val, test = split_by_date(panel, "As_Of_Date")

    grades_train = make_relevance_grades(train["Composite_Score"], train["Composite_Score"])
    grades_val = make_relevance_grades(train["Composite_Score"], val["Composite_Score"])
    grades_test = make_relevance_grades(train["Composite_Score"], test["Composite_Score"])
    train = train.assign(Relevance=grades_train)
    val = val.assign(Relevance=grades_val)
    test = test.assign(Relevance=grades_test)

    model = train_ltr_ranker(train, val)
    test_sorted = test.sort_values("Query_ID")
    test_sorted = test_sorted.assign(LTR_Score=score_ranker(model, test_sorted))

    ltr_groups = [
        (g["Relevance"].values, g["LTR_Score"].values)
        for _, g in test_sorted.groupby("Query_ID")
        if len(g) >= 2
    ]
    ltr_metrics = evaluate_ranking(ltr_groups, k=3, relevance_threshold=3)

    baseline_test_df = baseline_test_df.assign(Relevance=grades_test, Baseline_Score=baseline_test_preds)
    baseline_groups = [
        (g["Relevance"].values, g["Baseline_Score"].values)
        for _, g in baseline_test_df.groupby("Query_ID")
        if len(g) >= 2
    ]
    baseline_metrics = evaluate_ranking(baseline_groups, k=3, relevance_threshold=3)

    print(f"  Query groups evaluated: {len(ltr_groups)}")
    print("  Baseline-as-ranker:", {k: round(v, 3) for k, v in baseline_metrics.items()})
    print("  LTR (LambdaMART):  ", {k: round(v, 3) for k, v in ltr_metrics.items()})

    passed = ltr_metrics["NDCG@3"] >= baseline_metrics["NDCG@3"]
    print(f"Stage 4 result: {'PASS - LTR matches or beats baseline ranking' if passed else 'FAIL - LTR underperforms baseline'}\n")
    return model, passed


def stage5_submodels():
    print("=== STAGE 5: Sub-models (Price / Delivery / Quality / Risk) ===")
    hist = build_order_level_history_features(df, min_history=10)
    train, val, test = split_by_date(hist, "Order_Date")
    print(f"  train={len(train)} val={len(val)} test={len(test)}")

    results = {}

    for name, spec in SUBMODELS.items():
        model = train_submodel(spec, train, val)
        if spec.kind == "classification":
            best_threshold = find_best_threshold(model, val)
            metrics, _ = evaluate_submodel(spec, model, test, threshold=best_threshold)
            results[name] = metrics
            print(f"  {name}: AUC={metrics['AUC']:.3f} F1={metrics['F1']:.3f} (threshold={best_threshold})")
        else:
            metrics, _ = evaluate_submodel(spec, model, test)
            results[name] = metrics
            print(f"  {name}: MAE={metrics['MAE']:.3f} R2={metrics['R2']:.3f}")

    passed = (
        results["Price"]["R2"] > 0
        and results["Delivery"]["R2"] > 0
        and results["Quality"]["R2"] > 0
        and (np.isnan(results["Risk"]["AUC"]) or results["Risk"]["AUC"] > 0.5)
    )
    print(f"Stage 5 result: {'PASS' if passed else 'FAIL'}\n")
    return results, passed


if __name__ == "__main__":
    snapshot_dates = pd.date_range("2022-06-01", "2026-07-01", freq="W-MON")
    panel = build_snapshot_panel(df, snapshot_dates, min_orders=10)
    print(f"Snapshot panel built: {len(panel)} rows across {panel['Query_ID'].nunique()} query groups\n")

    baseline_model, baseline_test_df, baseline_test_preds, s3_pass = stage3_baseline(panel)
    ltr_model, s4_pass = stage4_ltr(panel, baseline_model, baseline_test_preds, baseline_test_df)
    submodel_results, s5_pass = stage5_submodels()

    panel.to_csv(os.path.join(os.path.dirname(__file__), "..", "Data", "processed", "part2_snapshot_panel.csv"), index=False)
    print("Overall Part 2:", "PASS" if (s3_pass and s4_pass and s5_pass) else "REVIEW NEEDED")
