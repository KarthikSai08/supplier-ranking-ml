import sys
import os
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from supplier_ranking.features.supplier_features import get_eligible_suppliers
from supplier_ranking.features.supplier_features import compute_supplier_features, compute_all_supplier_features
from supplier_ranking.features.scoring import compute_composite_score

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "Data", "raw", "supplier_ranking_dataset.csv")
AS_OF = "2026-01-01"

df = pd.read_csv(DATA_PATH)


def test_stage0_eligibility():
    print("=== STAGE 0: Eligibility filter ===")
    passed = True
    sample_products = df["Product_Name"].drop_duplicates().sample(5, random_state=1).tolist()  # type: ignore
    for product in sample_products:
        expected = sorted(
            df[(df["Product_Name"] == product) & (pd.to_datetime(df["Order_Date"]) < AS_OF)]
            ["Supplier_ID"].unique().tolist()  # type: ignore
        )
        actual = get_eligible_suppliers(df, product, AS_OF)
        ok = expected == actual
        passed = passed and ok
        print(f"  {product}: {len(actual)} eligible suppliers -> {'PASS' if ok else 'FAIL'}")
    print(f"Stage 0 result: {'PASS' if passed else 'FAIL'}\n")
    return passed


def test_stage1_features():
    print("=== STAGE 1: Point-in-time feature engineering ===")
    supplier_id = df["Supplier_ID"].iloc[0]
    computed = compute_supplier_features(df, supplier_id, AS_OF)

    manual_sub = df[(df["Supplier_ID"] == supplier_id) & (pd.to_datetime(df["Order_Date"]) < AS_OF)].copy()
    manual_on_time_pct = round((manual_sub["On_Time_Flag"] == "Yes").mean() * 100, 2)
    manual_total_orders = len(manual_sub)
    manual_rejection_pct = round(manual_sub["Rejected_Qty_Tons"].sum() / manual_sub["Delivered_Qty_Tons"].sum() * 100, 2)

    checks = {
        "Total_Orders": (computed["Total_Orders"] == manual_total_orders),
        "On_Time_Pct": (computed["On_Time_Pct"] == manual_on_time_pct),
        "Rejection_Pct": (computed["Rejection_Pct"] == manual_rejection_pct),
    }
    for k, ok in checks.items():
        print(f"  {k}: computed={computed[k]} manual={locals().get('manual_' + k.lower(), 'n/a')} -> {'PASS' if ok else 'FAIL'}")

    passed = all(checks.values())
    print(f"Stage 1 result: {'PASS' if passed else 'FAIL'}\n")
    return passed, computed


def test_stage2_scoring():
    print("=== STAGE 2: Composite score sanity check ===")
    features_df = compute_all_supplier_features(df, AS_OF)
    features_df = features_df[features_df["Total_Orders"] > 0]
    scored = compute_composite_score(features_df)  # type: ignore[arg-type]

    for category, group in scored.groupby("Product_Category"):
        top = group.sort_values("Composite_Score", ascending=False).head(3)
        bottom = group.sort_values("Composite_Score", ascending=True).head(3)
        print(f"  Category: {category}")
        print("  Top 3:")
        print(top[["Supplier_Name", "On_Time_Pct", "Rejection_Pct", "Quality_Pct", "Avg_Price_Per_Ton", "Composite_Score"]].to_string(index=False))
        print("  Bottom 3:")
        print(bottom[["Supplier_Name", "On_Time_Pct", "Rejection_Pct", "Quality_Pct", "Avg_Price_Per_Ton", "Composite_Score"]].to_string(index=False))
        top_avg_ontime = top["On_Time_Pct"].mean()
        bottom_avg_ontime = bottom["On_Time_Pct"].mean()
        ok = top["Composite_Score"].min() >= bottom["Composite_Score"].max()
        print(f"  Top scores all >= bottom scores: {'PASS' if ok else 'FAIL'}\n")

    return scored


if __name__ == "__main__":
    s0 = test_stage0_eligibility()
    s1, feats_sample = test_stage1_features()
    scored = test_stage2_scoring()
    scored.to_csv(os.path.join(os.path.dirname(__file__), "..", "Data", "processed", "part1_supplier_scores.csv"), index=False)
    print("Overall Part 1:", "PASS" if (s0 and s1) else "FAIL")
