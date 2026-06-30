# Prophet Models — Dengue & Chikungunya (State & City Level)

Code in this folder fits, validates, and submits the Prophet models used for this challenge entry. It reads the per-state files produced by `data_prep/agg_data_uf.r` (`processed_data/<disease>/<disease>_<UF>_agg.csv.gz`) and the per-city files produced by `data_prep/summarise_sel_cities.r` (`processed_data/<disease>/sel_cities/<disease>_<geocode>_agg.csv.gz`), and writes everything under `python_prophet/results/`.

This is a [uv](https://docs.astral.sh/uv/) project, independent of the Poetry `pyproject.toml` at the repository root (that one is a leftover from the IMDC challenge template and unused here).

## Files

| File | Role |
|---|---|
| `main.py` | Unused `uv init` stub left over from project scaffolding; not part of the pipeline. |
| `src/data_loader.py` | Loads a state's or city's processed table (`load_state()`, `load_city()`), adds the `log_cases_lag52` autoregressive feature, builds each fold's train/target split (`get_fold_data()`), and fills any missing target-window regressor with a training-only seasonal mean (`_fill_missing_regressors()`). Also defines `get_target_dates()`, which computes each fold's target-season length dynamically via the `epiweeks` package (52 or 53 weeks, depending on the year) rather than hardcoding 52. |
| `src/model.py` | Builds and fits the Prophet model (`_build_model()`, `fit_and_forecast()`), generates multi-quantile forecasts via a seasonal-stratified residual bootstrap (`_seasonal_bootstrap()`), and computes validation metrics — Weighted Interval Score, MAE, RMSE, and 50/80/90/95% interval coverage (`compute_metrics()`, `wis()`) — plus a basic output sanity check (`validate_output()`). |
| `src/run_all.py` | **Orchestration.** Loops `_process_unit()` (4 folds per unit, via `data_loader`/`model`) over all states or all focal cities for a disease, in parallel (`joblib.Parallel`), writing forecast CSVs and per-unit metrics CSVs. Resumable via a `concluded_units.csv` checkpoint per `(disease, scope)`. |
| `src/submit_predictions.py` | **Submission.** Uploads the forecast CSVs in `results/<disease>/<scope>/predictions/` to the Mosqlimate Predictions Registry via the `mosqlient` Python package, looking up each unit's admin codes from the matching `processed_data/` file. Retries transient HTTP 502/503/504 errors with backoff, and skips (rather than crashing) a unit that the API reports as already submitted ("Duplication found"). |
| `results/<disease>/<scope>/predictions/<unit>_target_<fold>.csv` | Forecast files — `date` plus `pred` and `lower_<level>`/`upper_<level>` at 50/80/90/95% — one per unit x fold (1–4). Fold 4 is the actual submission window. |
| `results/<disease>/<scope>/metrics/<unit>_metrics.csv` | Per-fold validation metrics (`wis`, `mae`, `rmse`, `coverage_50/80/90/95`, `n_obs`) for that unit, one row per fold. |
| `results/<disease>/<scope>/concluded_units.csv` | Checkpoint so `run_all.py` can be safely interrupted and resumed without redoing finished units. |

## How to run

From `python_prophet/`, with the data-prep pipeline (see the root README's "Data and Variables" section and `data_prep/README.md`) already run:

```bash
uv sync

# Fit + forecast every state, dengue (defaults: --disease dengue --scope state)
uv run python src/run_all.py

# Fit + forecast every city for a given disease (Optional City-Level Challenges)
uv run python src/run_all.py --scope city --disease dengue
uv run python src/run_all.py --scope city --disease chikungunya

# Chikungunya, state level
uv run python src/run_all.py --disease chikungunya

# Subset of units, more/less parallelism, or a clean rerun
uv run python src/run_all.py --states MG,SP --n-jobs 4
uv run python src/run_all.py --reset-checkpoint
```

Each invocation writes/updates `results/<disease>/<scope>/predictions/`, `results/<disease>/<scope>/metrics/`, and the resumability checkpoint `results/<disease>/<scope>/concluded_units.csv`.

Once forecasts exist, preview and submit them:

```bash
# Dry run — resolves every (unit, target) pair and prints what would be uploaded, without calling the API.
uv run python src/submit_predictions.py --dry-run

# Real submission for one disease/scope, all units with forecasts:
uv run python src/submit_predictions.py --disease dengue --scope city \
    --repository EzequielEBS/3rd_imdc_emap_epidematicos_prophet \
    --commit <commit_hash>

# A specific subset of units:
uv run python src/submit_predictions.py --disease chikungunya --scope state \
    --repository EzequielEBS/3rd_imdc_emap_epidematicos_prophet \
    --commit <commit_hash> --units AC,SP
```

## Credentials for `submit_predictions.py`

`submit_predictions.py` never hardcodes the Mosqlimate API key. Create a `.env` file in the **repository root** (already excluded from Git via `.gitignore`) with:

```
MOSQLIMATE_API_KEY=your-key-here
```

Get your key from the "Auth" section of your Mosqlimate profile at [mosqlimate.org](https://mosqlimate.org/). Also pass the `--repository` and `--commit` flags so they match the GitHub repository and commit hash you are actually submitting.

## Modeling summary

- Response: `log1p(cases)`, back-transformed with `expm1()` and clipped at zero.
- One independent Prophet model per (state or city) x disease, `growth="flat"` with built-in seasonality disabled in favor of a single explicit yearly term (`fourier_order=10`).
- Regressors (added with `standardize=True`): `temp_med_*_lag4`, `precip_med_*_lag4`, `enso`, `pdo`, `log_cases_lag52` — fixed per scope (see the root README's "Data and Variables" section for the exact state vs. city column names).
- Prediction intervals: seasonal-stratified residual bootstrap (1,000 paths, ±4-epiweek window), not Prophet's built-in intervals — see the root README's "Predictive Uncertainty" section.
- Validation: 4 train/target folds per unit, scored by Weighted Interval Score (and MAE/RMSE/coverage) in `model.compute_metrics()`.

See the root `README.md` for the full Team, Data, Model Training, Data Usage Restriction, Predictive Uncertainty, and References sections required by the IMDC submission format.
