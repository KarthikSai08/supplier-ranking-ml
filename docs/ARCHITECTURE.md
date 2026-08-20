# Architecture

This document explains how the system is put together so a new contributor can
orient themselves in 10 minutes. It is the map; the code is the territory.

## What the system does

Given a company's **order history** (`Data/raw/supplier_ranking_dataset.csv`),
the system:

1. **Trains models** that learn what "good" means from history (composite
   scoring, a regression model, a learning-to-rank model, and four sub-models).
2. **Answers new purchase orders (POs)** through two interactive flows:
   - `suggest` — a fast, filter-driven ranker (no training).
   - `recommend` — the full AI advisor: who should I buy this product from,
     and why?

Everything is **point-in-time** and **anti-leakage**: any value computed for
"as of date X" uses only orders strictly *before* X.

## The mental model: layers

```
┌───────────────────────────────────────────────────────────────────────┐
│  LAYER 4 - Entry points (what a user runs)                            │
│  app.py (argparse) ──► dispatch  (python -m supplier_ranking.app)     │
│    ├─ "panel" / "baseline" / "submodels" / "all"  → pipeline.runner   │
│    ├─ "suggest"   → app.run_suggest                                   │
│    └─ "advisor"   → app.run_advisor                                   │
├───────────────────────────────────────────────────────────────────────┤
│  LAYER 3 - Application services (orchestration, no I/O policy)        │
│  pipeline/runner.py     : snapshot panel, baseline+LTR eval,          │
│                           sub-models (via the SUBMODELS registry)     │
│  advisor.py             : eligibility → features → shrink → ML →      │
│                           blend → explain (the advisor pipeline)      │
│  app.py                 : the two PO flows (prompts + validation)     │
├───────────────────────────────────────────────────────────────────────┤
│  LAYER 2 - Domain logic (pure, testable, framework-light)             │
│  features/  : eligibility + supplier features, composite scoring,     │
│               order-level rolling features                            │
│  models/    : baseline regressor, LTR ranker, sub-models + registry   │
│  cold_start.py · weight_blend.py · explain.py  : Stage 6 / 7 / 8      │
├───────────────────────────────────────────────────────────────────────┤
│  LAYER 1 - Shared infrastructure (small, stable abstractions)         │
│  config.py  : every shared constant (split dates, paths, defaults)    │
│  io.py      : the ONLY place raw CSVs are read + column validation    │
│  evaluation/: metrics + time-based train/val/test splitting           │
└───────────────────────────────────────────────────────────────────────┘
```

**Rule of thumb for dependencies:** upper layers may import lower layers;
lower layers never import upper layers. `config.py`, `io.py`, and
`evaluation/splits.py` are depended on by many modules — change them
carefully.

## Stage → module map

| Stage | What happens | Where it lives |
|---|---|---|
| 0 | Eligibility: who supplied this product/category before the date? | `features/supplier_features.py` |
| 1 | Point-in-time supplier features (8 aggregates) | `features/supplier_features.py` |
| 2 | Composite score (0–100, per category, weighted) | `features/scoring.py` |
| — | Snapshot panel: stages 1+2 repeated weekly, stacked | `pipeline/runner.py` (`build_snapshot_panel`) |
| 3 | Baseline regressor predicts Composite_Score | `models/baseline_regressor.py` |
| 4 | LambdaMART learning-to-rank (grades 0–4) | `models/ltr_ranker.py` |
| 5 | Order-level sub-models: price / delivery / quality / risk | `models/submodels.py`; features in `features/order_history_features.py` |
| 6 | Cold-start: shrink low-history features toward the category mean | `cold_start.py` |
| 7 | Blend ML score with the user's priority weights | `weight_blend.py` |
| 8 | TreeSHAP contributions → plain-language "here's why" | `explain.py` |

## The two data paths

### Training/evaluation path (`pipeline/runner.py`)

```
raw CSV ──► snapshot_panel.build_snapshot_panel (weekly snapshots)
        ──► scoring.compute_composite_score (label)
        ──► baseline_regressor.train_baseline_regressor (XGBRegressor)
        ──► ltr_ranker.train_ltr_ranker (XGBRanker, relevance grades)
        ──► sub-models via models/registry (order-level rolling features)
        ──► evaluation/metrics.py (NDCG@k, MAP@k, ...)
```

### Advisor path (`advisor.py` + `app.run_advisor`)

```
product requirement
  → supplier_features.get_eligible_suppliers  (who has supplied it before?)
  → supplier_features.compute_all_supplier_features  (as of the placement date)
  → cold_start.shrink_features               (pull low-history suppliers toward category mean)
  → advisor._trained_baseline                (baseline regressor, cached per dataset)
  → weight_blend.user_weighted_score + blend_scores   (α·ML + (1−α)·User)
  → explain.explain_top_pick                 (TreeSHAP → sentence)
  → "USE: <supplier> — here's why"
```

## Where to put new code

| I want to add... | Put it in... |
|---|---|
| A new supplier feature | `features/supplier_features.py` + register in `FEATURE_COLUMNS` |
| A new weight/prior | `features/scoring.py` (`DEFAULT_WEIGHTS` / `HIGHER_IS_BETTER`) or `cold_start.py` (`SHRINKABLE_FEATURES`) |
| A new sub-model | one entry in `models/submodels.py` (`SUBMODELS` dict) — no pipeline changes needed |
| A new app subcommand | `app.py` (parser + dispatch) + a runner function in `pipeline/runner.py` |
| A new evaluation metric | `evaluation/metrics.py` |
| A new shared constant | `config.py` |

Full step-by-step recipes are in [PLAYBOOK.md](PLAYBOOK.md).

## Design conventions (SOLID)

The code is procedural (modules of functions), so SOLID is applied at module
granularity:

- **Single Responsibility** — one module, one job. `app.py` parses, dispatches,
  and owns the interactive flows; `pipeline/runner.py` does the pipeline work;
  `io.py` owns file access.
- **Open/Closed** — `models/submodels.py` (`SUBMODELS` registry) is the
  extension point for sub-models; adding one never requires editing the
  pipeline.
- **Liskov** — every sub-model goes through the same
  `train_submodel`/`evaluate_submodel` contract regardless of kind.
- **Interface Segregation** — modules import only what they need (e.g.
  `FEATURES` is shared, not dragged in wholesale).
- **Dependency Inversion** — high-level code depends on small abstractions
  (`config`, `io`, `splits`, the `SUBMODELS` registry), never on concrete
  CSV parsing or inline date-split logic.