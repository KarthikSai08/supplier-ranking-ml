# Supplier Ranking System — Cheatsheet

Full reference: flow, architecture, algorithms, methods, metrics, and the review plan.
Companion to `README.md`. This file is documentation only — it describes what the code does today.

---

## 1. Verdict on the intended flow

**Intended flow (your description):** user has requirements for a new PO → the system ranks suppliers against those requirements → the user *reconsiders* the ranked list using their own judgment → the user chooses the winner.

**Current state:** the `suggest` flow implements exactly this human-in-the-loop loop:

```
PO requirements (product, delivery date, price cap, quality/rejection/on-time minimums,
                 importance weights 0-10 for price/delivery/quality/on-time/rejection)
        │
        ▼
Hard filters (exclusions with reason) ──► Weighted match scoring (0-100) ──► Ranked suggestions
                                                                                  │
                                                                                  ▼
                                                          Human review: user compares and decides
```

So the flow itself is **correct for the suggestion + human-decision model**.

**Recommended, newer flow:** the `advisor` command is the full AI advisor. It chains the learned ML
models end-to-end:

```
User requirement: "Need 200t Iron Ore Fines, urgent"
        │
        ▼
eligibility.py      who has supplied this product before?        → Stage 0
        ▼
cold_start.py       features as of today, shrunk for newbies     → Stage 6
        ▼
trained baseline    ML score per eligible supplier (XGBoost)     → Stage 3, trained per dataset
        ▼
weight_blend.py     ML score blended with user's priorities (α)  → Stage 7
        ▼
explain.py (SHAP)   top pick + plain-language "here's why"       → Stage 8
        ▼
"Use Radhe Ore Metals Ltd — here's why"
```

The ML models that were siloed from the original `suggest` flow (baseline regressor, price/delivery/quality/risk sub-models, LambdaMART) are now used by `advisor`; `suggest` remains the lightweight, filter-driven quick view.

Secondary gaps that remain:
1. The learned sub-model forecasts (predicted price, delivery days, quality, risk for a supplier's *next* order) are not yet surfaced as recommendation signals.
2. Quantity is only used for cost estimation, not as a ranking input (volume discounts / capacity are ignored).
3. No persistence/serving layer yet (model is retrained per dataset and cached in-process via `lru_cache`).

---

## 2. End-to-end flow

```
                    ┌─────────────────────────  TRAINING / EVALUATION SIDE  ─────────────────────────┐
                    │                                                                                │
  Data/raw/supplier_ranking_dataset.csv                                                              │
        │  (order-level history: 20 columns, see §6)                                                 │
        ▼                                                                                            │
  STAGE 0  Eligibility filter (§3.1)          product + as_of_date -> suppliers with history < date │
        ▼                                                                                            │
  STAGE 1  Point-in-time features (§3.2)      8 supplier metrics computed from orders BEFORE date    │
        ▼                                                                                            │
  STAGE 2  Composite scoring (§3.3)           min-max per category -> weighted 0-100 score           │
        ▼                                                                                            │
  SNAPSHOT PANEL (§3.4)                        weekly snapshots (W-MON) stacked;                    │
                                              Query_ID = As_Of_Date_Product_Category                │
        ▼                                                                                            │
  STAGE 3  Baseline pointwise regressor (§3.5)   XGBRegressor  Composite_Score ~ 8 features          │
        ▼                                                                                            │
  STAGE 4  Learning-to-rank / LambdaMART (§3.6)  XGBRanker rank:ndcg, grades 0-4                     │
        ▼                                                                                            │
  STAGE 5  Sub-models (§3.7)                  price / delivery / quality / risk per order            │
        ▼                                                                                            │
  EVALUATION (§5)                             time-based train/val/test split, ranking metrics       │
                    └────────────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────────  SUGGESTION SIDE (interactive)  ─────────────────────┐
  User PO requirements (§3.8)  ──►  hard filters  ──►  weighted match score (0-100)  ──► ranked list │
                                                                                                      │
  ──► Human reviews (compares options, applies own judgment)  ──►  USER PICKS THE WINNER              │
                    └──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Algorithms & methods per stage

### 3.1 Stage 0 — Eligibility (features/supplier_features.py)
- `get_eligible_suppliers(df, product, as_of)`: suppliers with at least one order of that product strictly **before** `as_of_date` (point-in-time, no leakage).
- `get_eligible_suppliers_by_category(df, category, as_of)`: same, at category level.

### 3.2 Stage 1 — Supplier features (features/supplier_features.py)
For each supplier as of a date, computed only from orders with `Order_Date < as_of`:

| Feature | Formula |
|---|---|
| `Total_Orders` | order count before the date |
| `On_Time_Pct` | `mean(On_Time_Flag == "Yes") × 100` |
| `Rejection_Pct` | `Σ Rejected_Qty_Tons / Σ Delivered_Qty_Tons × 100` |
| `Quality_Pct` | mean of `Quality_Inspection_Score` |
| `Avg_Delivery_Days` | mean of `(Actual_Delivery_Date - Order_Date)` in days |
| `Avg_Price_Per_Ton` | mean of `Final_Price_Per_Ton` for that product (fallback: all the supplier's orders) |
| `Total_Purchase_Value` | `Σ (Final_Price_Per_Ton × Delivered_Qty_Tons)` |

`Avg_Price_Per_Ton` is the only **product-scoped** feature: when a product name is supplied, only that product's orders feed it. On-Time %, Rejection %, and Quality % are supplier-wide (general reliability traits, same reasoning as the Stage 5 sub-models). `Cancellation_Pct` was removed — the dataset has no source column, so it was always 0.0.

### 3.3 Stage 2 — Composite score (features/scoring.py)
- Min-max normalize each metric **within its product category** (`(x - min)/(max - min)`; constant column → 0.5).
- `higher_is_better`: On_Time_Pct, Quality_Pct ↑ ; Rejection_Pct, Avg_Delivery_Days, Avg_Price_Per_Ton ↓.
- Weighted sum → `Composite_Score` in 0–100:

| Metric | Weight |
|---|---|
| On_Time_Pct | 22% |
| Rejection_Pct | 22% |
| Quality_Pct | 22% |
| Avg_Delivery_Days | 17% |
| Avg_Price_Per_Ton | 17% |

### 3.4 Snapshot panel (pipeline/runner.py)
- Weekly Monday snapshots from `SNAPSHOT_START` to `SNAPSHOT_END` (in `config.py`).
- Per snapshot: Stage 1 + Stage 2, keep suppliers with `Total_Orders >= 10`.
- Rows stacked; `Query_ID = "<As_Of_Date>_<Product_Category>"` is the query group for ranking evaluation.

### 3.5 Stage 3 — Baseline pointwise regressor (models/baseline_regressor.py)
- `XGBRegressor`: 300 trees, `max_depth=4`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=1.0`, `random_state=42`.
- Features: the 8 supplier features → target `Composite_Score`.
- Metrics: **MAE, R2, Spearman** rank correlation.

### 3.6 Stage 4 — Learning-to-rank (models/ltr_ranker.py)
- Relevance grades 0–4: quantile-binning (`n_grades=5`) of `Composite_Score`; bin edges computed on the **train** distribution, applied to val/test.
- `XGBRanker` objective `rank:ndcg`: 300 trees, `max_depth=4`, `learning_rate=0.4`, `subsample=0.8`, `colsample_bytree=0.8`. Groups = `Query_ID`.
- Compared against the baseline regressor used as a ranker (baseline's predicted scores order the suppliers).
- Metrics: **NDCG@3, MAP@3, MRR, Precision@3, Recall@3** (relevance threshold 3, k=3).

### 3.7 Stage 5 — Order-level sub-models (models/submodels.py)
- Rolling history features (features/order_history_features.py): window 20 orders, `min_periods=5` (price: `min_periods=3`), lagged by one order (`shift(1)`):
  `Hist_Order_Count, Hist_On_Time_Pct, Hist_Rejection_Pct, Hist_Quality_Pct, Hist_Avg_Delivery_Days, Hist_Avg_Price_Per_Ton, Category_Code`
- Targets:

| Model | Target | Type |
|---|---|---|
| Price | `Target_Price` = `Final_Price_Per_Ton` | regression |
| Delivery | `Target_Delivery_Days` = lead time (days) | regression |
| Quality | `Target_Quality` = `Quality_Inspection_Score` | regression |
| Risk | `Target_Risk` = `Goods_Returned_Flag == "Yes"` OR `Rejection_Pct_Order > 5.0` | classification |

- Regression: `XGBRegressor` 250 trees, depth 4, lr 0.05 (shared `train_regressor` helper).
- Risk: `XGBClassifier` 250 trees, depth 4, lr 0.05, `scale_pos_weight = negatives/positives`, threshold swept 0.05–0.95 step 0.05 on **validation** maximizing **F1**.

## 3.8 Suggestion & advisor flows (app.py)

**`suggest` (lightweight, filter-driven):**
User prompts: data file, category, product name (optional), quantity (optional), order placement date (default today), required delivery date, max price/ton, min quality, max rejection %, min on-time %, then importance weights 0–10 for `price, delivery, quality, on_time, rejection`.

1. Build Stage-1 features for all suppliers as of the placement date; keep `Total_Orders >= 1`.
2. **Hard filters** (each logs how many suppliers were excluded and why):
   - `Avg_Price_Per_Ton <= max_price`
   - `Quality_Pct >= min_quality`
   - `Rejection_Pct <= max_rejection`
   - `On_Time_Pct >= min_on_time`
   - `Est_Delivery_Date <= required_delivery_date`, where `Est_Delivery_Date = as_of + Avg_Delivery_Days`
3. **Match score** for the survivors via `user_weighted_score` (Stage 7): per-dimension min-max normalization across survivors (inverted where lower-is-better), `Score = 100 × Σ(wᵢ·normᵢ)/Σ(wᵢ)`.
4. Sort descending → ranked suggestions table + estimated total cost for the cheapest option (qty × price).
5. Optional save to `po_suggestions_<date>_<category>.csv`.

**`advisor` (full AI advisor, Stage 0/1/6/3/7/8):**
1. Stage 0 eligibility: suppliers who supplied the **product** strictly before the placement date.
2. Stage 1 features as of the placement date, restricted to eligible suppliers — `Avg_Price_Per_Ton` is computed from the **product's** orders (falls back to all orders if none).
3. Stage 6 cold-start shrink toward the category average (`x_shrunk = (n·x + k·prior)/(n+k)`, k=5).
4. Stage 3 baseline regressor (XGBoost, trained per dataset, cached in-process) → `ML_Score`.
5. Stage 7 blend: `user_weighted_score` from the user's 0–10 priorities, then `Final = α·ML + (1−α)·User` (α default 0.5). "Urgent" orders force the delivery weight to ≥ 7.
6. Stage 8 TreeSHAP explanation of the top pick → "USE: <name> — <plain-language reason>".
7. Ranked shortlist (top 10) with final/ML/user scores, estimated delivery date, price, quality, rejection, on-time, order count + estimated total cost (qty × cheapest price).
8. Optional save to `po_recommend_<date>_<product>.csv`.

**Decision is always left to the human**: the ranked list is decision support; the user reconsiders and picks.

---

## 4. Architecture (module map)

```
src/supplier_ranking/
├── app.py                    entry point: panel | baseline | submodels | all | suggest | advisor
├── advisor.py                advisor pipeline: eligibility → cold start → ML → blend → explain (advise)
├── config.py                 single source of truth for shared constants
├── io.py                     data access: load_dataset + required-column validation
├── cold_start.py             Stage 6: shrink low-history features toward the category mean (k=5)
├── weight_blend.py           Stage 7: user_weighted_score + blend_scores (α = ML vs priorities)
├── explain.py                Stage 8: TreeSHAP contributions + plain-language "here's why"
├── features/
│   ├── supplier_features.py  Stages 0-1: eligibility + 8 supplier-level aggregates
│   ├── scoring.py            Stage 2: per-category min-max + weighted composite score
│   └── order_history_features.py  Stage 5: rolling 20-order history features + targets
├── models/
│   ├── baseline_regressor.py Stage 3 pointwise regressor (Composite_Score)
│   ├── ltr_ranker.py         Stage 4 LambdaMART (rank:ndcg)
│   └── submodels.py          Stage 5: price/delivery/quality/risk + SUBMODELS registry + shared helpers
├── pipeline/
│   └── runner.py             snapshot panel + stage runners (build_panel, run_baseline, run_ltr, run_submodels)
└── evaluation/
    ├── metrics.py            NDCG@k, MAP@k, MRR, Precision@k, Recall@k
    └── splits.py             time-based train/val/test splitting

tests/
├── test_part1.py             Stages 0-2 (eligibility, features, scoring) -> PASS
├── test_part2.py             Stages 3-5 (baseline, LTR, sub-models) -> PASS
└── test_part3.py             Stages 6-8 (cold start, blending, explain, advisor wiring) -> PASS

docs/                         all documentation (README, ARCHITECTURE, GLOSSARY, CONTRIBUTING, PLAYBOOK, CHEATSHEET)

Data/
├── raw/supplier_ranking_dataset.csv   input order history (committed)
└── processed/                         panel + test outputs (gitignored)
```

**Key invariants:**
- **Anti-leakage:** all features come only from orders strictly before the as-of/order date; rolling features are lagged by one order.
- **Time-based split:** train `< 2024-01-01`, val `2024-01-01..2026-01-01`, test `>= 2026-01-01`.
- Same 8 features feed Stage 1→3; a separate rolling-feature set feeds Stage 5.

---

## 5. Evaluation & reference numbers

| Stage | Metrics | Reference (bundled data) |
|---|---|---|
| 3 Baseline | MAE, R2, Spearman | test: MAE 8.42, R2 0.816, Spearman 0.960 |
| 4 LTR | NDCG@3, MAP@3, MRR, P@3, R@3 | LTR NDCG@3 0.836 vs baseline-as-ranker 0.758; MAP/MRR/P@3 1.000, R@3 0.240 |
| 5 Price | MAE, R2 | MAE 5705, R2 0.838 |
| 5 Delivery | MAE, R2 | MAE 2.80, R2 0.224 |
| 5 Quality | MAE, R2 | MAE 2.59, R2 0.511 |
| 5 Risk | AUC, F1 (threshold tuned on val) | AUC 0.760, F1 0.404, threshold 0.45 |

Test PASS criteria:
- `test_part1.py`: eligibility matches manual filter; features match manual computation; composite top scores ≥ bottom scores per category.
- `test_part2.py`: Stage 3 `R2 > 0.5 and Spearman > 0.5`; Stage 4 `LTR NDCG@3 >= baseline NDCG@3`; Stage 5 `R2 > 0` for price/delivery/quality and `AUC > 0.5` (or NaN) for risk.
- `test_part3.py`: cold-start shrinkage moves low-history suppliers toward the category prior and dampens snapshot volatility; raising the price weight improves the cheapest supplier's rank; TreeSHAP contributions sum to the model prediction and the reason is coherent; `advise` ranks only eligible suppliers with ML/User/Final scores.

Reference results (bundled data) for `advisor` on `Iron Ore Lumps` (as-of 2026-01-01, α=0.5, uniform priorities): 8 eligible suppliers; top pick **Radhe Ore Metals Ltd** (Final 83.5, ML 80.6, User 86.4) — "rejection rate 0.9 (better than the category average), quality 97.1 (better), on-time 76.0 (better)".

---

## 6. Data schema

`Data/raw/supplier_ranking_dataset.csv` (order-level):

`Order_ID, Supplier_ID, Supplier_Name, Product_Category, Product_Name, Order_Date, Expected_Delivery_Date, Actual_Delivery_Date, Delivery_Delay_Days, On_Time_Flag, Quoted_Price_Per_Ton, Final_Price_Per_Ton, Order_Qty_Tons, Delivered_Qty_Tons, Qty_Variation_Tons, Rejected_Qty_Tons, Rejection_Pct_Order, Quality_Inspection_Score, Goods_Returned_Flag, Return_Reason`

The `suggest` command validates the 12 columns it needs and reports any missing ones.

---

## 7. App command cheat sheet

```
python -m supplier_ranking.app all                    # panel + baseline + LTR + sub-models (no args needed)
python -m supplier_ranking.app panel                  # build Data/processed/snapshot_panel.csv  (~3 min)
python -m supplier_ranking.app baseline               # Stage 3 + 4 on the existing panel        (~1 min)
python -m supplier_ranking.app submodels              # Stage 5                                  (~1 min)
python -m supplier_ranking.app suggest                # interactive PO -> ranked suggestions     (~1 s)
python -m supplier_ranking.app advisor                # AI advisor: top pick + plain-language why (~3 min first run, cached)
python -m supplier_ranking.app suggest --data-path my.csv
```

| Option | Applies to | Default |
|---|---|---|
| `--data-path` | all (incl. suggest, advisor) | `Data/raw/supplier_ranking_dataset.csv` |

The snapshot range is fixed in `config.py` (`SNAPSHOT_START`/`SNAPSHOT_END`); `advisor` trains the baseline on weekly snapshots of that range (train < 2024, val 2024–2026) and caches it per dataset in-process; only the first run pays the ~3 min cost.

Run without install: `$env:PYTHONPATH = "src"; python -m supplier_ranking.app <stage>`.

---

## 8. Review plan — agreed target architecture (from user design) — IMPLEMENTED

Target flow for `advisor` (as designed by the user) — now live:

```
User requirement: "Need 200t Iron Ore Fines, urgent"
        │
        ▼
eligibility.py        Who has supplied this product before?            [src/supplier_ranking/features/supplier_features.py]
        ▼
cold_start.py         Features as of today, shrunk for new suppliers    [src/supplier_ranking/cold_start.py]
        ▼
trained baseline      ML score per eligible supplier                    [src/supplier_ranking/models/baseline_regressor.py,
regressor                                                               trained per dataset via advisor._trained_baseline]
        ▼
weight_blend.py       Blend with user's priorities this time            [src/supplier_ranking/weight_blend.py]
        ▼
explain.py (SHAP)     Top pick + plain-language reason                  [src/supplier_ranking/explain.py — XGBoost
                                                                        pred_contribs, no extra dependency]
        ▼
"Use Radhe Ore Metals Ltd — here's why"  (run_advisor in app.py)
```

Component status:

| Component | Status | Decision taken |
|---|---|---|
| `supplier_features.py` (eligibility) | done | restrict to suppliers who supplied the **product** strictly before the placement date (exact product name) |
| `cold_start.py` | done | `x_shrunk = (n·x + k·prior)/(n+k)`, prior = **category mean**, k = 5 (`DEFAULT_K`) |
| trained baseline regressor | done | **retrain per dataset** with in-process `lru_cache` (maxsize 8); persistence/serving deferred |
| `weight_blend.py` | done | `final = α·ML + (1−α)·user_score`; α default **0.5**, configurable per session; uniform weights if none given |
| `explain.py` | done | TreeSHAP `pred_contribs` → top 3 contributing features → "better/worse than the category average" sentence |
| output line | done | "USE: <name>" + reason + ranked shortlist (top 10) + cost estimate |

Legend: S = small (hours), M = medium (1–2 days).

Open items (not blocking):
1. Sub-model forecasts (price/delivery/quality/risk) not yet surfaced as recommendation signals.
2. Quantity is not yet a ranking input (volume discounts / capacity).
3. Model persistence/serving layer (P7) — training currently happens per process; caching is in-memory only.
4. Stage 4 LambdaMART is trained and evaluated but not used inside `advisor` (the pointwise baseline-as-ranker scores eligible suppliers).