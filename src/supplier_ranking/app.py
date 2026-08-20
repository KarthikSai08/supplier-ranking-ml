"""Interactive app and single entry point: python -m supplier_ranking.app <stage>."""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from supplier_ranking.advisor import advise
from supplier_ranking.config import (
    ADVISOR_MIN_ORDERS,
    DEFAULT_ALPHA,
    DEFAULT_DATA_PATH,
    DEFAULT_PANEL_OUTPUT,
    SNAPSHOT_END,
    SNAPSHOT_START,
)
from supplier_ranking.features.supplier_features import compute_all_supplier_features
from supplier_ranking.io import load_dataset
from supplier_ranking.pipeline.runner import build_panel, run_baseline, run_submodels
from supplier_ranking.weight_blend import user_weighted_score

MIN_ORDERS = ADVISOR_MIN_ORDERS

SUGGEST_DISPLAY_COLUMNS = [
    "Rank",
    "Supplier_Name",
    "Match_Score",
    "Avg_Price_Per_Ton",
    "Est_Delivery_Date",
    "On_Time_Pct",
    "Rejection_Pct",
    "Quality_Pct",
    "Total_Orders",
]

ADVISOR_DISPLAY_COLUMNS = [
    "Rank",
    "Supplier_Name",
    "Final_Score",
    "ML_Score",
    "User_Score",
    "Avg_Price_Per_Ton",
    "On_Time_Pct",
    "Rejection_Pct",
    "Quality_Pct",
    "Total_Orders",
]


def _prompt(text, default=None, validator=None, required=False):
    """Ask for input, re-prompting until an optional validator passes."""
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"{text}{suffix}: ").strip()
        value = raw if raw else default
        if value is None and required:
            print("  ! Please enter a value.")
            continue
        if validator is not None:
            error = validator(value)
            if error:
                print(f"  ! {error}")
                continue
        return value


def _validate_date(value):
    """Return an error message if value is not a parseable date, else None."""
    try:
        pd.Timestamp(value)
        return None
    except (ValueError, TypeError):
        return "Not a valid date (use YYYY-MM-DD)."


def _validate_float(value, minimum=None, maximum=None):
    """Return an error message if value is not a number in range, else None."""
    try:
        number = float(value)
    except (ValueError, TypeError):
        return "Not a valid number."
    if minimum is not None and number < minimum:
        return f"Must be at least {minimum}."
    if maximum is not None and number > maximum:
        return f"Must be at most {maximum}."
    return None


def _validate_file_exists(value):
    """Return an error message if value is not an existing file, else None."""
    if not Path(value).is_file():
        return f"File not found: {value}"
    return None


def _validate_category(df, value):
    """Return an error message if value is not a known category, else None."""
    if value not in df["Product_Category"].unique():
        return f"Category '{value}' not found. Available: {', '.join(sorted(df['Product_Category'].unique()))}"
    return None


def _validate_product(df, category, value):
    """Return an error message if value is not a product in category, else None."""
    available = df.loc[df["Product_Category"] == category, "Product_Name"].unique()
    if value and value not in available:
        return f"Product '{value}' not found in {category}. Available: {', '.join(sorted(available))}"
    return None


def _prompt_weights():
    """Ask for 0-10 importance per ranking dimension."""
    print("How important is each factor for THIS order? (0 = irrelevant, 10 = critical)")
    weights = {}
    for name, label in (
        ("price", "low price"),
        ("delivery", "fast delivery"),
        ("quality", "high quality"),
        ("on_time", "reliable on-time"),
        ("rejection", "low rejection"),
    ):
        weights[name] = float(
            _prompt(f"  Importance of {label}", "5", validator=lambda v: _validate_float(v, 0, 10))
        )
    return weights


def suggest_suppliers(
    df,
    as_of_date,
    category,
    product=None,
    required_delivery_date=None,
    max_price=None,
    min_quality=None,
    max_rejection=None,
    min_on_time=None,
    weights=None,
    min_orders=MIN_ORDERS,
):
    """Rank suppliers passing the hard filters, scored by the user's priority weights."""
    features = compute_all_supplier_features(df, as_of_date, product_name=product)
    features = features[features["Total_Orders"] >= min_orders].copy()

    if category:
        features = features[features["Product_Category"] == category].copy()
    if product:
        suppliers = df.loc[df["Product_Name"] == product, "Supplier_ID"].unique()
        features = features[features["Supplier_ID"].isin(suppliers)].copy()

    features["Est_Delivery_Date"] = (
        pd.Timestamp(as_of_date) + pd.to_timedelta(features["Avg_Delivery_Days"], unit="D")
    ).dt.strftime("%Y-%m-%d")

    filtered_out = 0

    def _apply(mask, label):
        nonlocal filtered_out
        if mask is None:
            return
        rejected = (~mask).sum()
        filtered_out += rejected
        if rejected:
            print(f"  - {rejected} supplier(s) excluded: {label}")
        features.drop(index=features.index[~mask], inplace=True)

    _apply(
        features["Avg_Price_Per_Ton"] <= max_price if max_price is not None else None,
        f"price over {max_price:,.0f}/ton" if max_price is not None else "price filter",
    )
    _apply(
        features["Quality_Pct"] >= min_quality if min_quality is not None else None,
        f"quality below {min_quality}",
    )
    _apply(
        features["Rejection_Pct"] <= max_rejection if max_rejection is not None else None,
        f"rejection over {max_rejection}%",
    )
    _apply(
        features["On_Time_Pct"] >= min_on_time if min_on_time is not None else None,
        f"on-time below {min_on_time}%",
    )
    if required_delivery_date is not None:
        _apply(
            pd.to_datetime(features["Est_Delivery_Date"]) <= pd.Timestamp(required_delivery_date),
            f"cannot deliver by {required_delivery_date}",
        )

    if features.empty:
        return features

    features["Match_Score"] = user_weighted_score(features, weights)
    features = features.sort_values("Match_Score", ascending=False).reset_index(drop=True)
    features.insert(0, "Rank", range(1, len(features) + 1))
    return features, filtered_out


def run_suggest(default_data_path: Path) -> None:
    """Prompt for PO requirements and print ranked supplier suggestions."""
    print()
    print("=== New Purchase Order - Supplier Suggestions ===")
    print("Enter your PO requirements and get ranked supplier suggestions.")
    print()

    data_path = _prompt("Path to your order history CSV", str(default_data_path), validator=_validate_file_exists, required=True)
    try:
        df = load_dataset(data_path)
    except ValueError as exc:
        print(f"\nERROR: {exc}")
        return

    categories = sorted(df["Product_Category"].unique())
    category = _prompt(
        "Product category",
        categories[0] if len(categories) == 1 else None,
        validator=lambda v: _validate_category(df, v),
        required=True,
    )
    product = _prompt("Product name (blank = whole category)", "", validator=lambda v: _validate_product(df, category, v))
    quantity = _prompt("Quantity (tons, blank = n/a)", "", validator=lambda v: _validate_float(v) if v else None)
    as_of_date = _prompt(
        "Order placement date (YYYY-MM-DD)",
        datetime.now().strftime("%Y-%m-%d"),
        validator=_validate_date,
    )
    required_delivery_date = _prompt(
        "Required delivery date (YYYY-MM-DD, blank = n/a)", "", validator=lambda v: _validate_date(v) if v else None
    )
    max_price = _prompt("Max price per ton (blank = n/a)", "", validator=lambda v: _validate_float(v) if v else None)
    min_quality = _prompt(
        "Min quality score 0-100 (blank = n/a)", "", validator=lambda v: _validate_float(v, 0, 100) if v else None
    )
    max_rejection = _prompt("Max rejection % (blank = n/a)", "", validator=lambda v: _validate_float(v, 0) if v else None)
    min_on_time = _prompt(
        "Min on-time % (blank = n/a)", "", validator=lambda v: _validate_float(v, 0, 100) if v else None
    )

    print()
    weights = _prompt_weights()

    print()
    print(f"Searching {len(df):,} orders as of {as_of_date}...")
    result = suggest_suppliers(
        df,
        as_of_date,
        category,
        product=product or None,
        required_delivery_date=required_delivery_date or None,
        max_price=float(max_price) if max_price else None,
        min_quality=float(min_quality) if min_quality else None,
        max_rejection=float(max_rejection) if max_rejection else None,
        min_on_time=float(min_on_time) if min_on_time else None,
        weights=weights,
    )

    ranked, filtered_out = result if isinstance(result, tuple) else (result, 0)
    if ranked.empty:
        print("\nNo suppliers match these requirements. Loosen the filters and try again.")
        return

    print()
    print(f"--- {category} - supplier suggestions ({len(ranked)} matches) ---")
    print(ranked[SUGGEST_DISPLAY_COLUMNS].to_string(index=False))
    if quantity:
        total = float(quantity) * ranked["Avg_Price_Per_Ton"]
        print(f"\nEstimated total cost for {quantity} tons (cheapest option): {total.min():,.2f}")

    if _prompt("Save suggestions to CSV", "n").lower() in ("y", "yes"):
        safe_category = str(category).lower().replace(" ", "_")
        out_path = Path(data_path).resolve().parent / f"po_suggestions_{as_of_date}_{safe_category}.csv"
        ranked.to_csv(out_path, index=False)
        print(f"Saved to {out_path}")

    print("\nDone.")


def run_advisor(default_data_path: Path) -> None:
    """Prompt for a new PO and print the AI advisor's top pick + ranked shortlist."""
    print()
    print("=== New PO - AI Supplier Advisor ===")
    print("The advisor learns from supplier history, applies your priorities,")
    print("and recommends the best supplier with a plain-language reason.")
    print()

    data_path = _prompt("Path to your order history CSV", str(default_data_path), validator=_validate_file_exists, required=True)
    try:
        df = load_dataset(data_path)
    except ValueError as exc:
        print(f"\nERROR: {exc}")
        return

    product = _prompt(
        "Product you are buying",
        required=True,
        validator=lambda v: (
            f"Product '{v}' not found. Available: {', '.join(sorted(df['Product_Name'].unique()))}"
            if v not in df["Product_Name"].unique()
            else None
        ),
    )
    quantity = _prompt("Quantity (tons, blank = n/a)", "", validator=lambda v: _validate_float(v) if v else None)
    urgent = _prompt("Is this an urgent order? (y/n)", "n")
    as_of_date = _prompt(
        "Order placement date (YYYY-MM-DD)",
        datetime.now().strftime("%Y-%m-%d"),
        validator=_validate_date,
    )

    weights = _prompt_weights()
    if urgent.lower() in ("y", "yes"):
        weights["delivery"] = max(float(weights["delivery"]), 7.0)
        print("  (urgent -> fast delivery importance raised to at least 7/10)")

    alpha = float(
        _prompt(
            "Trust the model vs your priorities (0 = only your priorities, 1 = only the model)",
            str(DEFAULT_ALPHA),
            validator=lambda v: _validate_float(v, 0, 1),
        )
    )

    print()
    print(f"Analyzing {len(df):,} orders as of {as_of_date}... (first run trains the model, ~3 min)")
    result = advise(
        data_path,
        product,
        as_of_date=as_of_date,
        quantity=quantity or None,
        weights=weights,
        alpha=alpha,
    )

    if result["top_pick"] is None:
        print(f"\n{result['reason']}")
        return

    ranked = result["ranked"]
    print()
    print(f"USE: {result['top_pick']['Supplier_Name']}")
    print(result["reason"])
    print()
    print(f"--- ranked shortlist ({len(ranked)} eligible suppliers) ---")
    print(ranked.head(min(10, len(ranked)))[ADVISOR_DISPLAY_COLUMNS].to_string(index=False))
    if quantity:
        total = float(quantity) * ranked["Avg_Price_Per_Ton"]
        print(f"\nEstimated total cost for {quantity} tons (cheapest option): {total.min():,.2f}")

    if _prompt("Save shortlist to CSV", "n").lower() in ("y", "yes"):
        safe_product = str(product).lower().replace(" ", "_")
        out_path = Path(data_path).resolve().parent / f"po_recommend_{as_of_date}_{safe_product}.csv"
        ranked.to_csv(out_path, index=False)
        print(f"Saved to {out_path}")

    print("\nDone.")


def main(argv=None) -> None:
    """Entry point: python -m supplier_ranking.app <stage> [--data-path ...]."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Supplier ranking app")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH, help="Path to raw order history CSV")
    subparsers = parser.add_subparsers(dest="stage", required=True)
    subparsers.add_parser("panel", help="Build the point-in-time snapshot panel")
    subparsers.add_parser("baseline", help="Train and evaluate the baseline regressor + LTR ranker")
    subparsers.add_parser("submodels", help="Train and evaluate price/delivery/quality/risk sub-models")
    subparsers.add_parser("all", help="Run the full pipeline (panel, baseline, LTR, sub-models)")
    subparsers.add_parser("suggest", help="Interactive: new-PO requirements -> ranked supplier suggestions")
    subparsers.add_parser("advisor", help="Interactive: AI advisor - top pick + plain-language reason")
    args = parser.parse_args(argv)

    if args.stage == "suggest":
        run_suggest(args.data_path)
        return

    if args.stage == "advisor":
        run_advisor(args.data_path)
        return

    if args.stage in ("panel", "all"):
        panel = build_panel(args.data_path, SNAPSHOT_START, SNAPSHOT_END)
    else:
        panel = None

    if args.stage in ("baseline", "all"):
        if panel is None:
            panel = pd.read_csv(DEFAULT_PANEL_OUTPUT)
        run_baseline(panel)

    if args.stage in ("submodels", "all"):
        run_submodels(args.data_path)


if __name__ == "__main__":
    main()