"""Unit tests: fast, isolated checks of every module on small synthetic data."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest

from supplier_ranking.advisor import advise
from supplier_ranking.cold_start import shrink_features
from supplier_ranking.evaluation.metrics import evaluate_ranking, ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank
from supplier_ranking.evaluation.splits import split_by_date
from supplier_ranking.features.scoring import DEFAULT_WEIGHTS, compute_composite_score, normalize_column
from supplier_ranking.features.supplier_features import (
    compute_all_supplier_features,
    compute_supplier_features,
    get_eligible_suppliers,
)
from supplier_ranking.io import load_dataset
from supplier_ranking.models.submodels import SUBMODELS
from supplier_ranking.weight_blend import blend_scores, user_weighted_score


def _orders_df() -> pd.DataFrame:
    rows = []

    def add(oid, sid, sname, prod, date, lead, on_time, qty, rej, qual, price):
        rows.append(
            {
                "Order_ID": oid,
                "Supplier_ID": sid,
                "Supplier_Name": sname,
                "Product_Category": "Raw Material",
                "Product_Name": prod,
                "Order_Date": date,
                "Actual_Delivery_Date": (pd.Timestamp(date) + pd.Timedelta(days=lead)).strftime("%Y-%m-%d"),
                "On_Time_Flag": on_time,
                "Delivered_Qty_Tons": qty,
                "Rejected_Qty_Tons": rej,
                "Quality_Inspection_Score": qual,
                "Final_Price_Per_Ton": price,
            }
        )

    for i, date in enumerate(pd.date_range("2022-07-04", periods=18, freq="60D")):
        add(
            f"S1-IO-{i}", "S1", "Good Steels", "Iron Ore", date.strftime("%Y-%m-%d"),
            10, "Yes" if i < 16 else "No", 100, 5 if i < 2 else 0, 95, 5000,
        )
    for i, date in enumerate(pd.date_range("2022-08-01", periods=14, freq="70D")):
        add(
            f"S2-IO-{i}", "S2", "Bad Steels", "Iron Ore", date.strftime("%Y-%m-%d"),
            25, "No" if i < 10 else "Yes", 100, 15, 70, 6500,
        )
    for i, date in enumerate(pd.date_range("2024-03-01", periods=3, freq="120D")):
        add(
            f"S3-IO-{i}", "S3", "New Steels", "Iron Ore", date.strftime("%Y-%m-%d"),
            12, "Yes", 100, 0, 90, 5500,
        )
    for i, date in enumerate(pd.date_range("2022-09-05", periods=4, freq="200D")):
        add(
            f"S1-CO-{i}", "S1", "Good Steels", "Coal", date.strftime("%Y-%m-%d"),
            8, "Yes", 100, 0, 92, 8000,
        )
    return pd.DataFrame(rows)


ORDERS = _orders_df()
AS_OF = "2025-12-31"


def test_load_dataset_accepts_valid_csv(tmp_path):
    path = tmp_path / "orders.csv"
    ORDERS.to_csv(path, index=False)
    loaded = load_dataset(path)
    assert list(loaded.columns) == list(ORDERS.columns)


def test_load_dataset_rejects_missing_columns(tmp_path):
    path = tmp_path / "orders.csv"
    ORDERS.drop(columns=["Final_Price_Per_Ton"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Final_Price_Per_Ton"):
        load_dataset(path)


def test_split_by_date_partitions():
    panel = pd.DataFrame(
        {
            "As_Of_Date": ["2023-06-01", "2024-06-01", "2025-06-01", "2026-06-01"],
            "value": [1, 2, 3, 4],
        }
    )
    train, val, test = split_by_date(panel)
    assert list(train["value"]) == [1]
    assert list(val["value"]) == [2, 3]
    assert list(test["value"]) == [4]


def test_normalize_column_higher_is_better():
    series = pd.Series([10.0, 20.0, 30.0])
    assert list(normalize_column(series, True)) == pytest.approx([0.0, 0.5, 1.0])


def test_normalize_column_lower_is_better():
    series = pd.Series([10.0, 20.0, 30.0])
    assert list(normalize_column(series, False)) == pytest.approx([1.0, 0.5, 0.0])


def test_normalize_column_constant_series():
    series = pd.Series([7.0, 7.0, 7.0])
    assert list(normalize_column(series, True)) == pytest.approx([0.5, 0.5, 0.5])


def test_composite_score_best_and_worst_suppliers():
    features = pd.DataFrame(
        {
            "Supplier_ID": ["A", "B"],
            "Product_Category": ["RM", "RM"],
            "On_Time_Pct": [100.0, 0.0],
            "Rejection_Pct": [0.0, 100.0],
            "Quality_Pct": [100.0, 0.0],
            "Avg_Delivery_Days": [1.0, 100.0],
            "Avg_Price_Per_Ton": [100.0, 1000.0],
        }
    )
    scored = compute_composite_score(features)
    assert scored.loc[0, "Composite_Score"] == pytest.approx(100.0)
    assert scored.loc[1, "Composite_Score"] == pytest.approx(0.0)


def test_composite_score_weights_sum_to_one():
    assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)


def test_eligible_suppliers_exact_product_and_strict_date():
    eligible = get_eligible_suppliers(ORDERS, "Iron Ore", AS_OF)
    assert eligible == ["S1", "S2", "S3"]
    assert get_eligible_suppliers(ORDERS, "Coal", AS_OF) == ["S1"]
    assert "S3" not in get_eligible_suppliers(ORDERS, "Iron Ore", "2024-02-01")


def test_eligible_suppliers_excludes_orders_on_as_of_date():
    assert "S1" in get_eligible_suppliers(ORDERS, "Iron Ore", "2022-07-05")
    assert "S1" not in get_eligible_suppliers(ORDERS, "Iron Ore", "2022-07-04")


def test_features_count_orders_strictly_before_as_of():
    features = compute_supplier_features(ORDERS, "S1", AS_OF)
    assert features["Total_Orders"] == 22
    assert features["On_Time_Pct"] == pytest.approx(round(20 / 22 * 100, 2))


def test_price_scoped_to_product_when_requested():
    all_features = compute_all_supplier_features(ORDERS, AS_OF)
    scoped = compute_all_supplier_features(ORDERS, AS_OF, product_name="Iron Ore")
    s1_all = all_features.loc[all_features["Supplier_ID"] == "S1", "Avg_Price_Per_Ton"].iloc[0]
    s1_scoped = scoped.loc[scoped["Supplier_ID"] == "S1", "Avg_Price_Per_Ton"].iloc[0]
    assert s1_all == pytest.approx((18 * 5000 + 4 * 8000) / 22)
    assert s1_scoped == pytest.approx(5000.0)


def test_all_features_include_expected_columns():
    features = compute_all_supplier_features(ORDERS, AS_OF)
    expected = {"Supplier_ID", "Supplier_Name", "Product_Category", "Total_Orders", "On_Time_Pct", "Rejection_Pct", "Quality_Pct", "Avg_Delivery_Days", "Avg_Price_Per_Ton", "Total_Purchase_Value"}
    assert expected.issubset(set(features.columns))


def test_shrink_features_formula_with_low_history():
    features = pd.DataFrame(
        {
            "Supplier_ID": ["A", "B"],
            "Product_Category": ["RM", "RM"],
            "Total_Orders": [1, 9],
            "On_Time_Pct": [100.0, 0.0],
            "Rejection_Pct": [0.0, 10.0],
            "Quality_Pct": [90.0, 50.0],
            "Avg_Delivery_Days": [5.0, 25.0],
            "Avg_Price_Per_Ton": [100.0, 300.0],
        }
    )
    shrunk = shrink_features(features, k=5)
    assert shrunk.loc[0, "On_Time_Pct"] == pytest.approx((1 * 100 + 5 * 50) / 6)
    assert shrunk.loc[1, "On_Time_Pct"] == pytest.approx((9 * 0 + 5 * 50) / 14)


def test_shrink_features_keeps_deep_history_nearly_unchanged():
    features = pd.DataFrame(
        {
            "Supplier_ID": ["A"],
            "Product_Category": ["RM"],
            "Total_Orders": [1000],
            "On_Time_Pct": [95.0],
            "Rejection_Pct": [1.0],
            "Quality_Pct": [90.0],
            "Avg_Delivery_Days": [10.0],
            "Avg_Price_Per_Ton": [500.0],
        }
    )
    shrunk = shrink_features(features, k=5)
    assert shrunk.loc[0, "On_Time_Pct"] == pytest.approx(95.0, abs=0.01)


def test_shrink_features_priors_are_per_category():
    features = pd.DataFrame(
        {
            "Supplier_ID": ["A", "B"],
            "Product_Category": ["RM1", "RM2"],
            "Total_Orders": [1, 1],
            "On_Time_Pct": [100.0, 0.0],
            "Rejection_Pct": [0.0, 10.0],
            "Quality_Pct": [90.0, 50.0],
            "Avg_Delivery_Days": [5.0, 25.0],
            "Avg_Price_Per_Ton": [100.0, 300.0],
        }
    )
    shrunk = shrink_features(features, k=5)
    assert shrunk.loc[0, "On_Time_Pct"] == pytest.approx(100.0)
    assert shrunk.loc[1, "On_Time_Pct"] == pytest.approx(0.0)


def test_user_weighted_score_price_only():
    features = pd.DataFrame(
        {
            "Avg_Price_Per_Ton": [100.0, 200.0],
            "On_Time_Pct": [80.0, 80.0],
            "Quality_Pct": [90.0, 90.0],
            "Avg_Delivery_Days": [10.0, 10.0],
            "Rejection_Pct": [2.0, 2.0],
        }
    )
    scores = user_weighted_score(features, {"price": 10})
    assert list(scores) == pytest.approx([100.0, 0.0])


def test_user_weighted_score_quality_only():
    features = pd.DataFrame(
        {
            "Avg_Price_Per_Ton": [100.0, 100.0],
            "On_Time_Pct": [80.0, 80.0],
            "Quality_Pct": [80.0, 90.0],
            "Avg_Delivery_Days": [10.0, 10.0],
            "Rejection_Pct": [2.0, 2.0],
        }
    )
    scores = user_weighted_score(features, {"quality": 10})
    assert list(scores) == pytest.approx([0.0, 100.0])


def test_user_weighted_score_stays_in_0_100_bounds():
    features = pd.DataFrame(
        {
            "Avg_Price_Per_Ton": [100.0, 200.0, 300.0],
            "On_Time_Pct": [50.0, 80.0, 95.0],
            "Quality_Pct": [60.0, 85.0, 99.0],
            "Avg_Delivery_Days": [30.0, 15.0, 5.0],
            "Rejection_Pct": [10.0, 3.0, 0.0],
        }
    )
    scores = user_weighted_score(features, {"price": 5, "on_time": 5, "quality": 5, "delivery": 5, "rejection": 5})
    assert scores.between(0, 100).all()


def test_blend_scores_formula():
    ml = pd.Series([80.0, 60.0])
    user = pd.Series([70.0, 50.0])
    assert list(blend_scores(ml, user, alpha=0.5)) == pytest.approx([75.0, 55.0])
    assert list(blend_scores(ml, user, alpha=1.0)) == pytest.approx([80.0, 60.0])
    assert list(blend_scores(ml, user, alpha=0.0)) == pytest.approx([70.0, 50.0])


def test_ndcg_perfect_ranking_is_one():
    relevance = [3, 2, 1, 0]
    scores = [4.0, 3.0, 2.0, 1.0]
    assert ndcg_at_k(relevance, scores, 3) == pytest.approx(1.0)


def test_ndcg_reversed_ranking_below_one():
    relevance = [3, 2, 1, 0]
    scores = [1.0, 2.0, 3.0, 4.0]
    assert ndcg_at_k(relevance, scores, 3) < 1.0


def test_precision_recall_mrr_on_toy_group():
    relevance = [3, 2, 0]
    scores = [4.0, 3.0, 1.0]
    assert precision_at_k(relevance, scores, 3, 3) == pytest.approx(1 / 3)
    assert recall_at_k(relevance, scores, 3, 3) == pytest.approx(1.0)
    assert reciprocal_rank(relevance, scores, 3) == pytest.approx(1.0)


def test_evaluate_ranking_metrics():
    groups = [([3, 2, 1, 0], [4.0, 3.0, 2.0, 1.0])]
    metrics = evaluate_ranking(groups, k=3, relevance_threshold=3)
    assert metrics["NDCG@3"] == pytest.approx(1.0)
    assert metrics["MRR"] == pytest.approx(1.0)
    assert metrics["Precision@3"] == pytest.approx(1 / 3)
    assert metrics["Recall@3"] == pytest.approx(1.0)


def test_submodels_registry_contents():
    assert set(SUBMODELS) == {"Price", "Delivery", "Quality", "Risk"}
    assert SUBMODELS["Price"].kind == "regression"
    assert SUBMODELS["Risk"].kind == "classification"
    assert SUBMODELS["Risk"].target == "Target_Risk"


def test_advise_end_to_end(tmp_path):
    path = tmp_path / "orders.csv"
    ORDERS.to_csv(path, index=False)
    result = advise(
        path,
        "Iron Ore",
        as_of_date=AS_OF,
        snapshot_range=("2022-06-01", "2025-12-31"),
        alpha=0.5,
    )
    assert result["eligible_count"] == 3
    assert result["top_pick"] is not None
    assert result["top_pick"]["Supplier_ID"] in {"S1", "S2", "S3"}
    ranked = result["ranked"]
    assert {"ML_Score", "User_Score", "Final_Score", "Rank"} <= set(ranked.columns)
    blended = 0.5 * ranked["ML_Score"] + 0.5 * ranked["User_Score"]
    assert np.allclose(blended, ranked["Final_Score"], atol=0.05)
    assert ranked["Final_Score"].is_monotonic_decreasing
    assert result["reason"]
    assert result["top_pick"]["Supplier_Name"] in result["reason"]


def test_advise_with_no_eligible_supplier(tmp_path):
    path = tmp_path / "orders.csv"
    ORDERS.to_csv(path, index=False)
    result = advise(path, "Uranium", as_of_date=AS_OF)
    assert result["top_pick"] is None
    assert result["eligible_count"] == 0
    assert result["ranked"].empty


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))