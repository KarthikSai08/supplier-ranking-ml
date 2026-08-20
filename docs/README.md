# Supplier Ranking System

A production-oriented supplier evaluation pipeline that ranks suppliers using point-in-time order history. It combines classical feature engineering with gradient-boosted scoring, a learning-to-rank (LambdaMART) model, and per-order sub-models for price, delivery, quality, and risk forecasting.

**New here? Start with [ARCHITECTURE.md](ARCHITECTURE.md), then [GLOSSARY.md](GLOSSARY.md).**
For conventions and the do-not-break rules, see [CONTRIBUTING.md](CONTRIBUTING.md); for step-by-step recipes, [PLAYBOOK.md](PLAYBOOK.md); for the full stage reference, [CHEATSHEET.md](CHEATSHEET.md).

## Table of Contents

- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Workflow](#workflow)
- [Data](#data)
- [Getting Started](#getting-started)
- [App Usage](#app-usage)
- [Run & Test (Manual)](#run--test-manual)
- [New PO - Supplier Suggestions](#new-po---supplier-suggestions)
- [AI Supplier Advisor (`advisor`)](#ai-supplier-advisor-advisor)
- [Metrics](#metrics)
- [Notes](#notes)

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.9+ (developed on 3.11) |
| Data handling | `pandas`, `numpy` |
| ML models | `xgboost` (XGBRegressor, XGBClassifier, XGBRanker) |
| Metrics | `scikit-learn` (MAE, R2, AUC, F1), `scipy` (Spearman rank correlation) |
| Packaging | `setuptools` with src-layout, `pyproject.toml`, console script entry point |
| CLI | `argparse` (no external deps) |

Dependencies are declared in both `pyproject.toml` and `requirements.txt`:

- `numpy>=1.24`
- `pandas>=2.0`
- `scikit-learn>=1.3`
- `scipy>=1.10`
- `xgboost>=2.0`

## Project Structure

```
supplier_ranking_system/
├── src/supplier_ranking/
│   ├── __init__.py
│   ├── app.py                       # Entry point + interactive app: run_suggest + run_advisor (prompts and validation)
│   ├── app.py                       # Interactive app: run_suggest + run_advisor (prompts and validation)
│   ├── advisor.py                   # Advisor pipeline: eligibility -> cold start -> ML -> blend -> explain
│   ├── config.py                    # Single source of truth for all shared constants
│   ├── io.py                        # Data access: load_dataset with required-column validation
│   ├── cold_start.py                # Stage 6: shrink low-history features toward the category mean
│   ├── weight_blend.py              # Stage 7: user priority weights blended with the ML score (alpha)
│   ├── explain.py                   # Stage 8: TreeSHAP contributions -> plain-language "here's why"
│   ├── features/
│   │   ├── supplier_features.py     # Stages 0-1: eligibility + supplier-level aggregation features
│   │   ├── scoring.py               # Stage 2: weighted composite score (0-100, per category)
│   │   └── order_history_features.py# Stage 5: order-level rolling history features + targets
│   ├── models/
│   │   ├── baseline_regressor.py    # Stage 3: pointwise composite-score regressor
│   │   ├── ltr_ranker.py            # Stage 4: LambdaMART (rank:ndcg) ranker
│   │   └── submodels.py             # Stage 5: price/delivery/quality/risk + SUBMODELS registry
│   ├── pipeline/
│   │   └── runner.py                # Orchestration: snapshot panel + stage runners
│   └── evaluation/
│       ├── metrics.py               # NDCG@k, MAP@k, MRR, Precision@k, Recall@k
│       └── splits.py                # Time-based train/val/test splitting
├── tests/
│   ├── test_part1.py                # Stage 0-2: eligibility, features, scoring
│   ├── test_part2.py                # Stage 3-5: baseline, LTR, sub-models
│   └── test_part3.py                # Stage 6-8: cold start, blending, explain, advisor wiring
├── docs/                            # All documentation (see the docs/ folder)
│   ├── README.md                    # This file
│   ├── ARCHITECTURE.md              # Layers, data flow, module map
│   ├── GLOSSARY.md                  # Plain-English terms
│   ├── CONTRIBUTING.md              # Conventions, invariants, review checklist
│   ├── PLAYBOOK.md                  # Step-by-step recipes
│   └── CHEATSHEET.md                # Full stage-by-stage reference
├── Data/
│   ├── raw/                         # supplier_ranking_dataset.csv (input, committed)
│   └── processed/                   # Generated panels/scores (gitignored)
├── pyproject.toml
├── requirements.txt
└── .gitignore
```

## Workflow

The pipeline mirrors a staged, audit-friendly ML process:

### Stage 0 — Eligibility filtering
Given a product/category and an `as_of_date`, only suppliers with **order history strictly before the date** are eligible (strict point-in-time to prevent leakage).

### Stage 1 — Point-in-time feature engineering
For each supplier as of a snapshot date (`Data/raw/supplier_ranking_dataset.csv`):

| Feature | Meaning |
|---|---|
| `Total_Orders` | Order count before the snapshot date |
| `On_Time_Pct` | % of orders delivered on time |
| `Rejection_Pct` | Rejected tons / delivered tons |
| `Quality_Pct` | Mean quality inspection score |
| `Avg_Delivery_Days` | Mean lead time (actual delivery − order date) |
| `Avg_Price_Per_Ton` | Mean final price per ton (product-scoped when a product is given; otherwise all the supplier's orders) |
| `Total_Purchase_Value` | Σ price × delivered tons |

`Cancellation_Pct` was removed: the dataset has no source column for it, so it was always 0.0 and added no signal.

### Stage 2 — Composite scoring
Min-max normalized per **product category**, then weighted into a 0–100 `Composite_Score`:

| Metric | Weight |
|---|---|
| On_Time_Pct | 22% |
| Rejection_Pct | 22% |
| Quality_Pct | 22% |
| Avg_Delivery_Days | 17% |
| Avg_Price_Per_Ton | 17% |

### Snapshot panel
Repeats stages 1–2 across a weekly schedule of snapshot dates and stacks results into a panel. Each row is identified by `Query_ID = As_Of_Date_Product_Category`, which becomes the query group for ranking evaluation.

### Stage 3 — Baseline pointwise regressor
`XGBRegressor` (300 trees, depth 4, lr 0.05, L2 regularization) predicts `Composite_Score` from the 8 supplier features. Evaluated with **MAE, R2, and Spearman rank correlation**.

### Stage 4 — Learning-to-rank (LambdaMART)
`XGBRanker` with `rank:ndcg` (300 trees, depth 4, lr 0.2). Relevance grades (0–4) are derived by quantile-binning the composite score. Evaluated with **NDCG@3, MAP@3, MRR, Precision@3, Recall@3**, compared against the baseline used as a ranker.

### Stage 5 — Sub-models
Order-level rolling history features (20-order window, min 5 periods; 3 for price) predict per-order targets:

| Model | Target | Type |
|---|---|---|
| Price | `Target_Price` = final price/ton | Regression |
| Delivery | `Target_Delivery_Days` = lead time | Regression |
| Quality | `Target_Quality` = inspection score | Regression |
| Risk | `Target_Risk` = goods returned OR rejection > 5% | Classification |

The risk model uses `scale_pos_weight` (class imbalance) and tunes the decision threshold on validation to maximize F1.

### Train / validation / test split
All stages are time-based to avoid leakage:
- Train: before `2024-01-01`
- Validation: `2024-01-01` – `2026-01-01`
- Test: from `2026-01-01`

## Data

**Input:** `Data/raw/supplier_ranking_dataset.csv` — order-level records with columns:

`Order_ID, Supplier_ID, Supplier_Name, Product_Category, Product_Name, Order_Date, Expected_Delivery_Date, Actual_Delivery_Date, Delivery_Delay_Days, On_Time_Flag, Quoted_Price_Per_Ton, Final_Price_Per_Ton, Order_Qty_Tons, Delivered_Qty_Tons, Qty_Variation_Tons, Rejected_Qty_Tons, Rejection_Pct_Order, Quality_Inspection_Score, Goods_Returned_Flag, Return_Reason`

**Outputs (written to `Data/processed/`, gitignored):**
- `part1_supplier_scores.csv` — per-supplier composite scores (Stage 1–2)
- `part2_snapshot_panel.csv` — stacked snapshot panel (Stage 3–5 input)
- `snapshot_panel.csv` — panel written by the CLI `panel` stage

## Getting Started

```bash
# 1. (Recommended) create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate    # Linux/macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install the package (optional; not needed if you set PYTHONPATH instead)
pip install -e .
```

Verify the install:

```bash
python --version                 # 3.9+ (developed on 3.11)
python -m supplier_ranking.app --help
```

Without step 3 you can still run the app module directly:

```bash
set PYTHONPATH=src    # Windows PowerShell: $env:PYTHONPATH = "src"
python -m supplier_ranking.app --help
```

## App Usage

**Quickest start — run the whole pipeline with one command (no arguments needed):**

```bash
python -m supplier_ranking.app all
```

**Individual stages:**

```bash
# Build the point-in-time snapshot panel
python -m supplier_ranking.app panel

# Train/evaluate baseline regressor + LTR ranker (needs the panel first)
python -m supplier_ranking.app baseline

# Train/evaluate price/delivery/quality/risk sub-models
python -m supplier_ranking.app submodels

# New PO: enter requirements, get ranked supplier suggestions
python -m supplier_ranking.app suggest

# AI advisor: top pick + plain-language "here's why" (trains the model on first run)
python -m supplier_ranking.app advisor
```

The only option (applies to every stage; it has a default, so plain `python -m supplier_ranking.app all` also works):

| Option | Default | Description |
|---|---|---|
| `--data-path` | `Data/raw/supplier_ranking_dataset.csv` | Raw order history CSV |

The panel is always written to `Data/processed/snapshot_panel.csv`, and suppliers need at least 10 historical orders to be included. Note: model evaluation is time-split at `2024-01-01` / `2026-01-01`; the snapshot range is fixed in `config.py` (`SNAPSHOT_START` / `SNAPSHOT_END`).

## Run & Test (Manual)

Approximate runtimes on the bundled dataset (measured on a typical laptop):

| Command | What it does | Runtime |
|---|---|---|
| `python -m supplier_ranking.app panel` | Builds the snapshot panel → `Data/processed/snapshot_panel.csv` | ~3 min |
| `python -m supplier_ranking.app baseline` | Trains baseline regressor + LTR ranker on the existing panel | ~1 min |
| `python -m supplier_ranking.app submodels` | Trains price/delivery/quality/risk sub-models | ~1 min |
| `python -m supplier_ranking.app all` | Runs everything end-to-end (panel → baseline → LTR → sub-models) | ~5 min |
| `python -m supplier_ranking.app suggest` | New-PO flow: enter requirements, get ranked supplier suggestions | ~1 s |
| `python -m supplier_ranking.app advisor` | AI advisor: top pick + plain-language reason (trains/caches the baseline on first run) | ~3 min first run, then instant |
| `python tests/test_part1.py` | Stages 0–2: eligibility, features, composite scoring | ~10 s |
| `python tests/test_part2.py` | Stages 3–5: baseline, LTR, sub-models (rebuilds the panel internally) | ~4 min |
| `python tests/test_part3.py` | Stages 6–8: cold start, blending, explainability, advisor wiring | ~1 min |

**Step-by-step manual run:**

```bash
# 1. Build the panel (needed before the baseline stage)
python -m supplier_ranking.app panel

# 2. Train/evaluate baseline + LTR ranker (reads the panel from step 1)
python -m supplier_ranking.app baseline

# 3. Train/evaluate the four sub-models
python -m supplier_ranking.app submodels
```

**Manual test run:**

```bash
python tests/test_part1.py      # expect: Overall Part 1: PASS
python tests/test_part2.py      # expect: Overall Part 2: PASS
python tests/test_part3.py      # expect: Overall Part 3: PASS
```

`test_part2.py` builds the panel in memory (it does not reuse `Data/processed/snapshot_panel.csv`), so it is the most time-consuming step. `test_part3.py` trains a baseline on a shortened snapshot range for speed. The test scripts print per-stage results and write `part1_supplier_scores.csv` / `part2_snapshot_panel.csv` to `Data/processed/`.

**Reference output (test_part2.py, Stages 3–5):**

```
Stage 3 result: PASS           # Test: MAE=8.42 R2=0.816 Spearman=0.960
Stage 4 result: PASS - LTR matches or beats baseline ranking
                               # Baseline-as-ranker NDCG@3=0.758, LTR NDCG@3=0.836
Stage 5 result: PASS           # Price R2=0.84, Delivery R2=0.22, Quality R2=0.51, Risk AUC=0.76
```

## New PO - Supplier Suggestions

`python -m supplier_ranking.app suggest` is the mode for **creating a new purchase order**: you enter your PO requirements and it suggests suppliers ranked by how well they fit, based on their order history as of today (or any as-of date).

```bash
python -m supplier_ranking.app suggest                                    # prompts for everything
python -m supplier_ranking.app suggest --data-path my_orders.csv          # skip the file-path prompt
```

Prompts (all have defaults or can be left blank):

| Prompt | What it does |
|---|---|
| Path to your order history CSV | Your data file (required columns are validated) |
| Product category / product name | Restricts suggestions to the PO's product (name optional) |
| Quantity (tons) | Used to estimate total cost of the cheapest suggestion |
| Order placement date | Only orders strictly before this date count (default: today) |
| Required delivery date | Hard filter: suppliers that cannot deliver in time are excluded |
| Max price per ton | Hard filter |
| Min quality score (0–100) | Hard filter |
| Max rejection % | Hard filter |
| Min on-time % | Hard filter |
| Importance of low price / fast delivery / high quality / reliable on-time / low rejection (0–10) | **Your priorities** — they weight the ranking of the suppliers that passed the hard filters |

How it works: suppliers failing any hard requirement are excluded (with a message saying why), then the survivors are scored 0–100 by min-max normalizing each factor and weighting it by your stated importance. The result is a ranked suggestion list with expected delivery date and estimated total cost.

Example session:

```bash
$ python -m supplier_ranking.app suggest --data-path my_orders.csv
Path to your order history CSV [my_orders.csv]:
Product category: Raw Material
Product name (blank = whole category) []: Iron Ore
Quantity (tons, blank = n/a) []: 100
Order placement date (YYYY-MM-DD) [2026-08-20]: 2025-06-01
Required delivery date (YYYY-MM-DD, blank = n/a) []: 2025-06-15
Max price per ton (blank = n/a) []: 5500
Min quality score 0-100 (blank = n/a) []: 85
Max rejection % (blank = n/a) []:
Min on-time % (blank = n/a) []:

How important is each factor for THIS order? (0 = irrelevant, 10 = critical)
  Importance of low price [5]: 8
  Importance of fast delivery [5]: 5
  Importance of high quality [5]: 3
  Importance of reliable on-time [5]: 2
  Importance of low rejection [5]: 4

Searching 12 orders as of 2025-06-01...
  - 1 supplier(s) excluded: price over 5,500/ton
  - 1 supplier(s) excluded: cannot deliver by 2025-06-15

--- Raw Material - supplier suggestions (2 matches) ---
 Rank Supplier_Name  Match_Score  Avg_Price_Per_Ton Est_Delivery_Date  On_Time_Pct  Rejection_Pct  Quality_Pct  Total_Orders
    1 Acme Minerals        59.09            5076.67        2025-06-11        66.67           0.61        96.33             3
    2  Beta Traders        40.91            4650.00        2025-06-12        66.67           3.02        90.00             3

Estimated total cost for 100 tons (cheapest option): 465,000.00
Save suggestions to CSV [n]: y
Saved to my_orders/po_suggestions_2025-06-01_raw_material.csv
```

## AI Supplier Advisor (`advisor`)

`python -m supplier_ranking.app advisor` is the **full AI advisor** for a new purchase order. Unlike `suggest` (which applies your hard filters to a static scoring formula), it chains the trained ML models end-to-end:

```
product requirement → eligibility (who supplied it before?)
→ point-in-time features, cold-start shrunk toward the category average
→ baseline ML score per eligible supplier (XGBoost, trained on your dataset)
→ blended with your priority weights (alpha = model vs. your priorities)
→ TreeSHAP explanation → "USE: <supplier> — here's why"
```

```bash
python -m supplier_ranking.app advisor                                   # prompts for everything
python -m supplier_ranking.app advisor --data-path my_orders.csv         # skip the file-path prompt
```

Prompts:

| Prompt | What it does |
|---|---|
| Path to your order history CSV | Your data file (required columns are validated) |
| Product you are buying | Exact product name (validated against the data; eligible suppliers are those with history of it before the placement date) |
| Quantity (tons) | Used to estimate total cost of the cheapest shortlisted option |
| Is this an urgent order? (y/n) | Yes forces the "fast delivery" priority to at least 7/10 |
| Order placement date | Only orders strictly before this date count (default: today) |
| Importance of low price / fast delivery / high quality / reliable on-time / low rejection (0–10) | **Your priorities** for this order |
| Trust the model vs your priorities (0–1) | Blend alpha: 0 = only your priorities, 1 = only the ML model (default 0.5) |

How it works: only suppliers who have actually supplied the product before the placement date are considered; their features are computed as of that date — price per ton scoped to that specific product, the rest supplier-wide — then shrunk toward the category average (cold start — a supplier with little history is not penalized or over-trusted); the trained baseline regressor scores each one (`ML_Score`); your priorities produce a `User_Score`; the final ranking is `alpha × ML + (1 − alpha) × User`. The top pick is explained with a plain-language reason built from TreeSHAP feature contributions, e.g. *"rejection rate of 0.9 (better than the category average of 1.9); quality score of 97.1 (better than the category average of 93.9)"*.

The model is trained per dataset on the first run (weekly snapshots, ~3 min on the bundled data) and cached in-process, so repeated runs are instant.

Example session (bundled data):

```bash
$ python -m supplier_ranking.app advisor
=== New PO - AI Supplier Advisor ===
...
Product you are buying: Iron Ore Lumps
Quantity (tons, blank = n/a) []: 100
Is this an urgent order? (y/n) [n]: n
Order placement date (YYYY-MM-DD) [2026-08-20]: 2026-01-01
...
Trust the model vs your priorities (0 = only your priorities, 1 = only the model) [0.5]: 0.5

Analyzing 11,161 orders as of 2026-01-01... (first run trains the model, ~3 min)

USE: Radhe Ore Metals Ltd
Radhe Ore Metals Ltd is the top-ranked pick, mainly due to: rejection rate of 0.9 (better
than the category average of 1.9); quality score of 97.1 (better than the category average
of 93.9); on-time delivery rate of 76.0 (better than the category average of 59.4).

--- ranked shortlist (8 eligible suppliers) ---
 Rank Supplier_Name  Final_Score   ML_Score              User_Score        Avg_Price_Per_Ton  ...
    1  Radhe Ore Metals Ltd        83.49 80.550003       86.42             8776.78  ...
    2  Krishna Metal & Co          72.56 79.720001       65.39             8949.64  ...
    3  Hanuman Ore Traders         72.47 80.930000       64.00             9070.52  ...
...
Estimated total cost for 100 tons (cheapest option): 815,519.96
```

## Metrics

Reference run on the bundled dataset (sub-model stage):

| Model | Metric | Value |
|---|---|---|
| Price | R2 | 0.84 |
| Price | MAE | ~5705 ₹/ton |
| Delivery | R2 | 0.22 |
| Quality | R2 | 0.51 |
| Risk | AUC | 0.76 (threshold 0.45) |

Ranking stages report NDCG@3, MAP@3, MRR, Precision@3, and Recall@3 per query group (`Query_ID`). Reference comparison on the test split:

| Ranker | NDCG@3 | MAP@3 | MRR | Precision@3 | Recall@3 |
|---|---|---|---|---|---|
| Baseline-as-ranker | 0.758 | 1.000 | 1.000 | 1.000 | 0.240 |
| LTR (LambdaMART) | 0.836 | 1.000 | 1.000 | 1.000 | 0.240 |

## Notes

- **Anti-leakage:** every feature is computed strictly from orders before the snapshot date; evaluation splits are time-based.
- **Tests:** `tests/test_part1.py`, `test_part2.py`, and `test_part3.py` cover stages 0-2 (eligibility, features, scoring), 3-5 (baseline, LTR, sub-models), and 6-8 (cold start, weight blending, explainability, advisor wiring). They import from the `supplier_ranking` package and read `Data/raw/supplier_ranking_dataset.csv`. Run with `python tests/test_part1.py` / `python tests/test_part2.py` / `python tests/test_part3.py` (or `pip install -e .` first, or set `PYTHONPATH=src`). See [Run & Test](#run--test-manual) for expected output.
- **Cheatsheet & roadmap:** `CHEATSHEET.md` contains the full reference — flow diagram, all algorithms per stage, module map, metrics, and the agreed advisor architecture (Stages 6-8 + `advisor`), now implemented. Open items: sub-model forecasts as ranking signals, quantity as a ranking input, and model persistence/serving.
- The `Data/` directory keeps its original casing from the initial layout; all code references it consistently. Both `Data/processed/` and `data/processed/` are gitignored.