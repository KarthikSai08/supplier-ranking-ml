import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from supplier_ranking.cold_start import shrink_features
from supplier_ranking.weight_blend import DIMENSIONS, user_weighted_score
from supplier_ranking.explain import explain_top_pick, shap_contributions, feature_priors
from supplier_ranking.advisor import advise, _trained_baseline
from supplier_ranking.features.supplier_features import compute_all_supplier_features, get_eligible_suppliers
from supplier_ranking.models.baseline_regressor import FEATURES as BASELINE_FEATURES

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "Data", "raw", "supplier_ranking_dataset.csv")
SHORT_SNAPSHOT_RANGE = ("2022-06-01", "2024-02-01")

df = pd.read_csv(DATA_PATH)


def _constructed_frame(noisy_q, noisy_r):
    rng = np.random.default_rng(7)
    bulk = pd.DataFrame(
        {
            "Supplier_ID": [f"B{i}" for i in range(20)],
            "Product_Category": "A",
            "Total_Orders": 100.0,
            "Quality_Pct": rng.normal(85, 2, 20),
            "Rejection_Pct": rng.normal(4, 0.5, 20),
            "Avg_Price_Per_Ton": 20000.0,
            "Avg_Delivery_Days": 10.0,
            "On_Time_Pct": 90.0,
        }
    )
    low = pd.DataFrame(
        [
            {"Supplier_ID": "L1", "Product_Category": "A", "Total_Orders": 2.0, "Quality_Pct": noisy_q[0], "Rejection_Pct": noisy_r[0], "Avg_Price_Per_Ton": 25000.0, "Avg_Delivery_Days": 12.0, "On_Time_Pct": 60.0},
            {"Supplier_ID": "L2", "Product_Category": "A", "Total_Orders": 5.0, "Quality_Pct": noisy_q[1], "Rejection_Pct": noisy_r[1], "Avg_Price_Per_Ton": 18000.0, "Avg_Delivery_Days": 8.0, "On_Time_Pct": 80.0},
        ]
    )
    return pd.concat([bulk, low], ignore_index=True)


def test_stage6_cold_start():
    print("=== STAGE 6: Cold-start shrinkage (deterministic properties) ===")
    a = _constructed_frame([40.0, 90.0], [20.0, 5.0])
    b = _constructed_frame([60.0, 70.0], [15.0, 3.0])
    sa = shrink_features(a).set_index("Supplier_ID")
    sb = shrink_features(b).set_index("Supplier_ID")
    prior_q = float(a.loc[a["Product_Category"] == "A", "Quality_Pct"].mean())

    ok1 = abs(sa.loc["L1", "Quality_Pct"] - prior_q) < abs(40.0 - prior_q)
    mature_raw = float(a.loc[a["Supplier_ID"] == "B0", "Quality_Pct"].iloc[0])
    ok2 = abs(sa.loc["B0", "Quality_Pct"] - mature_raw) < 0.5
    print(f"  low-history supplier pulled toward category prior -> {'PASS' if ok1 else 'FAIL'}")
    print(f"  mature supplier (100 orders) barely moved -> {'PASS' if ok2 else 'FAIL'}")

    raw_vol = abs(a.loc[a["Supplier_ID"].isin(["L1", "L2"]), "Quality_Pct"]
                  - b.loc[b["Supplier_ID"].isin(["L1", "L2"]), "Quality_Pct"].values).mean()
    shrunk_vol = abs(sa.loc[["L1", "L2"], "Quality_Pct"] - sb.loc[["L1", "L2"], "Quality_Pct"]).mean()
    ok3 = shrunk_vol < raw_vol
    print(f"  snapshot volatility: raw={raw_vol:.2f} shrunk={shrunk_vol:.2f} -> {'PASS' if ok3 else 'FAIL'}")
    feats = compute_all_supplier_features(df, "2026-01-01")
    low = feats[feats["Total_Orders"] <= 10].copy()  # type: ignore[index]
    print(f"  real dataset has only {len(low)} cold-start suppliers as of 2026 -> {'PASS' if len(low) <= 3 else 'FAIL'}")
    return ok1 and ok2 and ok3 and len(low) <= 3


def test_stage7_weight_blend():
    print("=== STAGE 7: Weight blending ===")
    feats = compute_all_supplier_features(df, "2026-01-01")
    feats = feats[feats["Product_Category"] == "Raw Material"]
    uniform = {name: 1.0 for name in DIMENSIONS}
    price_heavy = {"price": 0.70, "on_time": 0.10, "quality": 0.10, "delivery": 0.05, "rejection": 0.05}

    s_uniform = user_weighted_score(feats, uniform)  # type: ignore[arg-type]
    s_price = user_weighted_score(feats, price_heavy)  # type: ignore[arg-type]

    cheapest_idx = feats["Avg_Price_Per_Ton"].idxmin()  # type: ignore[attr-defined]
    rank_uniform = int((s_uniform > s_uniform.loc[cheapest_idx]).sum() + 1)
    rank_price = int((s_price > s_price.loc[cheapest_idx]).sum() + 1)
    print(f"  cheapest supplier rank: uniform weights -> #{rank_uniform}, price-heavy -> #{rank_price}")
    ok = rank_price <= rank_uniform
    print(f"  raising price weight moves the cheapest supplier up -> {'PASS' if ok else 'FAIL'}\n")
    return ok


def test_stage8_explain():
    print("=== STAGE 8: Explainability (TreeSHAP -> plain language) ===")
    result = advise(
        DATA_PATH, "Iron Ore Lumps", as_of_date="2026-01-01", snapshot_range=SHORT_SNAPSHOT_RANGE
    )
    model = _trained_baseline(DATA_PATH, SHORT_SNAPSHOT_RANGE)
    feats = compute_all_supplier_features(df, "2026-01-01")
    priors = feature_priors(feats)
    row = result["ranked"].iloc[0]

    contribs = shap_contributions(model, row[BASELINE_FEATURES].astype(float).to_frame().T)[0]
    pred = model.predict(row[BASELINE_FEATURES].astype(float).to_frame().T)[0]
    additive = np.isclose(contribs.sum(), pred, atol=1e-3)

    reason = explain_top_pick(model, row[BASELINE_FEATURES], priors, row["Supplier_Name"])
    ok = additive and row["Supplier_Name"] in reason and "category average" in reason
    print(f"  reason: {reason}")
    print(f"  SHAP additive property -> {'PASS' if additive else 'FAIL'}; reason coherent -> {'PASS' if ok else 'FAIL'}\n")
    return ok


def test_recommend_wiring():
    print("=== Recommend wiring: eligibility -> cold start -> ML -> blend -> explain ===")
    result = advise(
        DATA_PATH,
        "Iron Ore Lumps",
        as_of_date="2026-01-01",
        weights={name: 1.0 for name in DIMENSIONS},
        alpha=0.5,
        snapshot_range=SHORT_SNAPSHOT_RANGE,
    )
    assert result["top_pick"] is not None
    ranked = result["ranked"]
    top = result["top_pick"]

    ok1 = top["Supplier_Name"] == ranked.iloc[0]["Supplier_Name"] and ranked.iloc[0]["Rank"] == 1
    ok2 = "category average" in result["reason"]
    eligible = set(get_eligible_suppliers(df, "Iron Ore Lumps", "2026-01-01"))
    ok3 = set(ranked["Supplier_ID"]).issubset(eligible)
    ok4 = {"ML_Score", "User_Score", "Final_Score"}.issubset(ranked.columns)

    print(f"  top pick = {top['Supplier_Name']} (Final={top['Final_Score']}, ML={top['ML_Score']}, User={top['User_Score']})")
    print(f"  reason: {result['reason']}")
    print(f"  eligible suppliers: {result['eligible_count']}")
    print(f"  rank/score columns -> {'PASS' if ok1 and ok4 else 'FAIL'}")
    print(f"  explainable reason -> {'PASS' if ok2 else 'FAIL'}")
    print(f"  only eligible suppliers -> {'PASS' if ok3 else 'FAIL'}")
    return ok1 and ok2 and ok3 and ok4


if __name__ == "__main__":
    s6 = test_stage6_cold_start()
    s7 = test_stage7_weight_blend()
    s8 = test_stage8_explain()
    wiring = test_recommend_wiring()
    print("Overall Part 3:", "PASS" if (s6 and s7 and s8 and wiring) else "FAIL")
