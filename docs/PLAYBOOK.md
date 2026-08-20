# Playbook

Step-by-step recipes for the most common tasks. Each recipe lists the exact
files to touch and how to verify you did not break anything.

## Recipe 1 — Add a new supplier feature

1. Add the computation in `src/supplier_ranking/features/supplier_features.py`
   inside `compute_supplier_features()` (aggregate from orders before the
   as-of date — never leak).
2. Register the column name in `FEATURE_COLUMNS`.
3. If it should influence the composite score: add a weight in
   `features/scoring.py` (`DEFAULT_WEIGHTS`, weights must sum to 1.0) and a
   direction in `HIGHER_IS_BETTER`.
4. If it should be cold-start shrunk: add it to `cold_start.py`
   (`SHRINKABLE_FEATURES`) — and note `explain.py` `FEATURE_LABELS` /
   `LOWER_IS_BETTER` for the reason sentence.
5. Verify: `python tests/test_part1.py` and `python tests/test_part3.py`.

## Recipe 2 — Add a new sub-model

1. Add a target column in `features/order_history_features.py`
   (`target_cols` + a line computing it).
2. Register it in `models/submodels.py` `SUBMODELS`:
   `"MyModel": SubModel(target="Target_MyTarget", kind="regression")`
   (or `"classification"`).
3. That's it — `pipeline/runner.py` and the tests pick it up automatically.
4. Verify: `python tests/test_part2.py` (Stage 5 section).

## Recipe 3 — Add a new app subcommand

1. Add the subcommand in `src/supplier_ranking/app.py` `main()`.
2. Add the orchestration function in `pipeline/runner.py`.
3. Dispatch to it in `app.py` `main()`.
4. Verify: `python -m supplier_ranking.app --help` shows it; run it on the
   bundled data.

## Recipe 4 — Add a test

1. Match the stage to a file: Stages 0-2 → `tests/test_part1.py`,
   3-5 → `tests/test_part2.py`, 6-8 + advisor → `tests/test_part3.py`.
2. Import from the `supplier_ranking` package (tests set `PYTHONPATH`).
3. Print a `PASS`/`FAIL` line per check, and return a boolean.
4. Wire the new test into the `if __name__ == "__main__":` block.
5. Verify: run the test file end-to-end.

## Recipe 5 — Change the train/validation/test cutoffs

1. Edit `TRAIN_END` / `VAL_END` in `src/supplier_ranking/config.py` — one
   place, used everywhere (pipeline + advisor).
2. Re-run all three test files and re-record the reference numbers in
   `docs/CHEATSHEET.md` and `docs/README.md`.

## Recipe 6 — Point the system at a different dataset

1. `python -m supplier_ranking.app panel --data-path my.csv` (or the
   `--data-path` flag on any stage; `suggest`/`advisor` also prompt for it
   interactively).
2. Required columns are validated by `io.load_dataset` — a clear error is
   printed listing any missing ones.
3. Note: the advisor trains and caches the baseline **per
   dataset** (in-process), so the first run with a new file takes ~3 min.

## Recipe 7 — Understand why the advisor picked a supplier

1. `python -m supplier_ranking.app advisor`
2. Answer the prompts (product, quantity, date, priorities).
3. The reason sentence comes from `explain.py`: the top-3 TreeSHAP
   contributions of the winning supplier, phrased "better/worse than the
   category average". The score is `Final = 0.5·ML + 0.5·User` by default —
   set the blend alpha (0-1) at the prompt to trust the model more or less.

## Recipe 8 — Clean up generated files

```powershell
Remove-Item Data\processed\*.csv   # panels + test outputs, gitignored
```

`Data/processed/` and `Data/raw/po_*.csv` are generated outputs and safe to
delete; `Data/raw/supplier_ranking_dataset.csv` is the input and is NOT.