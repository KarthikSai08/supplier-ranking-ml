# Glossary

Plain-English definitions of the terms used across the codebase and docs.
Read this before reading the code; it removes most of the "what does this
mean?" friction.

## Core concepts

| Term | Meaning |
|---|---|
| **Order history / raw dataset** | `Data/raw/supplier_ranking_dataset.csv` — one row per order: who supplied what, when, at what price/quality/rejection/on-time. The single input to everything. |
| **Point-in-time** | Every number is computed from orders **strictly before** a date (the as-of/placement date). Nothing "future" leaks into a decision. |
| **Eligible supplier** | A supplier that has actually supplied the requested product (or category) before the as-of date. In `suggest`/`advisor`, only eligible suppliers are considered. |
| **As-of date** | The date "as of which" we look at history. Defaults to today for interactive flows; fixed dates are used for training snapshots. |
| **Snapshot** | One point-in-time view of all suppliers' features + composite scores at a weekly date. |
| **Snapshot panel** | All snapshots stacked into one table. Each row is one supplier at one snapshot; `Query_ID = As_Of_Date_Product_Category` groups rows for ranking evaluation. |
| **Composite score** | A 0–100 "how good is this supplier overall?" number: min-max normalized features (per category), weighted (22/22/22/17/17), × 100. It is the **training label** — the model learns to predict it. |
| **Feature** | A numeric property of a supplier/order, e.g. `On_Time_Pct`, `Avg_Price_Per_Ton`. |
| **Min-max normalization (norm)** | Rescale a column to 0–1: `(x − min) / (max − min)`. Lower-is-better metrics are inverted (`1 − norm`). A constant column becomes 0.5. |
| **Cold-start / shrinkage** | A supplier with little history is unreliable: pull its features toward the category average: `x_shrunk = (n·x + k·prior) / (n + k)`, where `n` = order count, `prior` = category mean, `k` = strength (default 5). A supplier with 100 orders barely moves; one with 2 orders is pulled close to the category. |
| **ML_Score** | The trained baseline regressor's prediction of the composite score for a supplier (captures nonlinear interactions the fixed formula misses). |
| **User_Score** | 0–100 score from your priority weights: min-max norm each dimension across the candidate set, weight by your importance (0–10), `100 × Σ(w·norm)/Σ(w)`. |
| **Alpha (α)** | The blend ratio: `Final = α·ML + (1−α)·User`. `α=0` = only your priorities, `α=1` = only the model, default 0.5. |
| **Urgent order** | In `advisor`, answering "yes" to urgent forces the *fast delivery* weight to at least 7/10 before scoring. |
| **Product-scoped price** | `Avg_Price_Per_Ton` uses only the ordered product's orders (when a product is given); all other features are supplier-wide. |
| **Relevance grade** | 0–4 buckets made by quantile-binning the composite score (edges from the training split). Used to train/evaluate the ranking model. |
| **Query group** | A group of suppliers that are compared together for ranking (one snapshot × one category). |
| **NDCG@k / MAP@k / MRR / Precision@k / Recall@k** | Ranking-quality metrics comparing the model's order vs the ideal order, evaluated at top-k (default 3). |
| **TreeSHAP** | A technique that attributes each feature's contribution to a model's prediction per row. XGBoost computes it natively (`pred_contribs`), so no extra package is needed. |

## Scores at a glance

| Score | Who sets the weights | Normalized across | Role |
|---|---|---|---|
| `Composite_Score` | The system (fixed weights) | Suppliers in the same category | Training label for the baseline model |
| `Match_Score` (`suggest`) | The user (0–10) | Suppliers that passed the hard filters | Ranks the `suggest` shortlist |
| `ML_Score` (`advisor`) | The model | — | Learned estimate of the composite score |
| `User_Score` (`advisor`) | The user (0–10) | Eligible suppliers | Your priorities |
| `Final_Score` (`advisor`) | α blend | — | The final ranking key |

## Pipeline stage names

| Stage | Name | One-line job |
|---|---|---|
| 0 | Eligibility | Who has supplied this before the date? |
| 1 | Supplier features | Aggregate each supplier's history into a feature row |
| 2 | Composite scoring | Turn features into a weighted 0–100 label |
| 3 | Baseline regressor | Learn to predict the composite score |
| 4 | Learning-to-rank | Learn to order suppliers within a query group |
| 5 | Sub-models | Predict the next order's price / delivery / quality / risk |
| 6 | Cold start | Shrink unreliable (low-history) features toward the category |
| 7 | Weight blending | Combine the ML score with your priorities |
| 8 | Explainability | Say *why* the top pick won, in plain language |