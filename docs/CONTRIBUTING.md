# Contributing

Everything a new contributor needs to get set up, run the project, and make
changes without breaking existing features.

## Setup (5 minutes)

```powershell
# 1. (Recommended) virtual environment
python -m venv .venv
.venv\Scripts\activate            # Windows PowerShell

# 2. Dependencies
pip install -r requirements.txt

# 3. Install the package (optional; not needed if you set PYTHONPATH instead)
pip install -e .

# 4. Verify
python --version                  # 3.9+ (developed on 3.11)
python -m supplier_ranking.app --help
```

**No install? No problem** — run the module directly:

```powershell
$env:PYTHONPATH = "src"           # Windows PowerShell
python -m supplier_ranking.app --help
```

## How to know you haven't broken anything (always do this)

Run the three test suites. They must all end with PASS:

```powershell
python tests/test_part1.py        # Stages 0-2  (~10 s)
python tests/test_part3.py        # Stages 6-8 + advisor wiring (~1 min)
python tests/test_part2.py        # Stages 3-5, rebuilds the panel (~4 min)
```

The **reference numbers** you should see (bundled data): test MAE 8.42,
R2 0.816, Spearman 0.960; LTR NDCG@3 0.836 vs baseline-as-ranker 0.758;
Price R2 0.838, Delivery R2 0.224, Quality R2 0.511, Risk AUC 0.760.
If these change significantly, either you changed model behavior on purpose,
or something regressed.

For a quick app sanity check:

```powershell
$env:PYTHONPATH = "src"
python -m supplier_ranking.app --help          # all 6 subcommands listed
python -m supplier_ranking.app suggest --data-path Data/raw/supplier_ranking_dataset.csv
```

## Coding conventions

- **Python 3.9+**, `snake_case` functions/variables, `UPPER_SNAKE_CASE`
  constants.
- **One module = one job.** New logic goes in the right layer
  (see [ARCHITECTURE.md](ARCHITECTURE.md) — the "where to put new code" table).
- **Every module starts with a one-line docstring** saying what it does and
  which stage it belongs to. Match the existing style.
- **Shared constants live in `config.py`, not inline.** Never redefine a date
  split or path in a new file.
- **Never read a CSV directly** — use `io.load_dataset()` so required-column
  validation is not skipped.
- **Never reimplement date splitting** — use `evaluation/splits.split_by_date`.
- **`type: ignore` comments** are used bare; the reason is obvious from
  context.
- **Tests** for new behavior belong in the matching `tests/test_partN.py`
  file (or a new file following the same style).

## The invariants — DO NOT BREAK THESE

These rules are load-bearing. If a change violates one, rethink the change:

1. **Anti-leakage.** Every feature is computed from orders strictly before the
   as-of/order date; rolling features are lagged by one order (`shift(1)`).
   Never use current/future data to score a supplier at a date.
2. **Eligibility is an exact product-name match** (`df["Product_Name"] ==
   product`) with orders strictly before the as-of date. Changing this changes
   every recommendation.
3. **Composite weights sum to 1.0** (`0.22+0.22+0.22+0.17+0.17`) and the
   `HIGHER_IS_BETTER` flags stay correct for each feature.
4. **Blend formula** stays `Final = α·ML + (1−α)·User` with α ∈ [0,1]
   (default 0.5 in `config.DEFAULT_ALPHA`).
5. **Urgent orders floor the delivery weight** at ≥ 7 (`max(w, 7)`) before
   scoring.
6. **Cold-start formula** stays `x_shrunk = (n·x + k·prior)/(n+k)` with
   default `k=5` (`cold_start.DEFAULT_K`).
7. **Time-based split** stays train `< 2024-01-01`, val through
   `2026-01-01`, test from there — unless you intentionally re-cut history,
   which also means re-running and re-recording every reference number.
8. **App command contract.** The six subcommands (`panel, baseline,
   submodels, all, suggest, advisor`) and the `--data-path` flag are the
   public interface — keep them stable.

## Review checklist (before you finish a change)

- [ ] `python tests/test_part1.py`, `test_part2.py`, `test_part3.py` all PASS.
- [ ] Reference numbers match the values above (or changed deliberately).
- [ ] No new duplication of constants/CSV reads/date splits.
- [ ] New modules have a 1-2 line docstring; comments are single lines.
- [ ] If you touched the advisor flow, the reason sentence still reads
      sensibly (run the app or `test_part3.py`).
- [ ] You did not commit secrets, generated files, or `Data/processed/`
      (gitignored).