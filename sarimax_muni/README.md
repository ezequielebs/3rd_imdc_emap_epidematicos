# SARIMAX Models — Dengue & Chikungunya (Municipality Level)

Code in this folder fits, selects, and submits the city-level SARIMAX models used for this challenge entry, for a fixed set of focal cities defined in `cities_dengue` (15 geocodes) and `cities_chikungunya` (10 geocodes) — these lists are repeated in `src/model_sel.r`, `src/fit.r`, and `src/sub_pred.r`. It reads the per-city files produced by `data_prep/sel_cities.r` + `data_prep/summarise_sel_cities.r` (`processed_data/dengue/sel_cities/dengue_<geocode>_agg.csv.gz`, `processed_data/chikungunya/sel_cities/chikungunya_<geocode>_agg.csv.gz`) and writes everything under `sarimax/results/`.

## Files

| File | Role |
|---|---|
| `src/utils.r` | Shared library: `fit_sarimax()` (core SARIMAX fit + forecast), covariate-selection helpers (`get_candidates()`, `filter_low_variance()`, `filter_by_correlation()`, `filter_redundant_indices()`, `select_best_per_variable()`, `build_covariate_combinations()`, `pca_all()`), the grid-search engine (`run_grid_search()`), and the scoring helper (`compute_metrics()`, computing WIS/MAE/RMSE/MAPE/coverage). Sourced by every other script. |
| `src/model_sel.r` | **Training / model selection.** Defines `run_model_selection()` (which also supports an optional state-level `state`/`concluded_states` path against `agg_data_uf.r`'s output, unused by this script's own loop) and loops it over `cities_dengue` / `cities_chikungunya` for both diseases, writing per-city CV metrics and a `best_wis_<disease>_all_cities.csv` summary. |
| `src/fit.r` | **Final fit & forecast generation.** For each city in `cities_dengue` / `cities_chikungunya`, refits the selected best model on all four train/target splits and writes one forecast CSV per city x target window to `results/preds/`. Both diseases have: (1) a retry loop that re-fits with the next-best formula/order (from `metrics_all_formulas_<disease>_<geocode>.csv`) for any city whose initial fit produced an actionable warning (convergence, NaN SEs), a hard `Arima()` error, or a **degenerate forecast** (`is_degenerate_forecast()`: every point prediction for a target window is exactly 0 — typical of sparse, low-incidence cities where the `log1p(cases)` forecast drifts ≤ 0 and gets clipped); (2) a `tryCatch()` around every `Arima()`-calling fit so a hard numerical failure (e.g. "non-finite finite-difference value") is recorded like a warning instead of crashing the script; and (3) persisted logs of every warning/error/degenerate-forecast flag from both the initial and retry passes to `results/metrics/warnings_<disease>.csv`, plus the formula/order that ultimately resolved each retried city to `results/metrics/resolved_formulas_<disease>.csv`. Also defines `fit_sarimax_epiweek()`, the variant used for the real submission window (split 4 — see the root README's "Data Usage Restriction" section). |
| `src/sub_pred.r` | **Submission.** Reads the per-city forecast CSVs in `results/preds/` and uploads them to the Mosqlimate Predictions Registry via the `mosqlient` Python package (through `reticulate`), with `adm_level = 2` (municipality), `adm_1` = the city's `uf_code`, and `adm_2` = its `geocode`. |
| `results/concluded_cities_dengue.csv`, `results/concluded_cities_chikungunya.csv` | Checkpoints so `model_sel.r` can be safely interrupted and resumed without redoing finished cities. |
| `results/metrics/` | One `metrics_all_formulas_<disease>_<geocode>.csv` per city/disease (every formula x order combination tried, ranked by mean CV metric); `best_wis_<disease>_all_cities.csv` (the winning combination per city); `mean_wis_best_model_<disease>_<geocode>.csv` (final per-split metrics of the chosen model); `warnings_<disease>.csv` (every `Arima()` warning and caught error from both the initial and retry passes of `fit.r`, tagged `phase = "initial"`/`"retry"`); `resolved_formulas_<disease>.csv` (which formula/order row eventually produced a clean fit for each retried city). |
| `results/preds/` | `pred_<disease>_<geocode>_target_<n>.csv` — the actual forecast files (date, point prediction, and 50/80/90/95% interval bounds), ready to submit. |

## How to run

From the repository root, with R and the packages listed in the root README's "Libraries and Dependencies" section installed:

```r
# Training / model selection for every focal city, both diseases. Safe to
# interrupt and re-run: already-finished cities are skipped via
# results/concluded_cities_*.csv.
source("sarimax/src/model_sel.r")

# Refit the winning model per city, retry on warnings/errors, and generate
# the 4 forecast windows. Logs warnings/errors to results/metrics/warnings_*.csv.
source("sarimax/src/fit.r")

# Submit the forecasts to Mosqlimate (see "Credentials" below).
source("sarimax/src/sub_pred.r")
```

To experiment with a single city without running the full loop, call `run_model_selection()` directly after sourcing `src/utils.r` and `src/model_sel.r`'s function definition, e.g.:

```r
source("sarimax/src/utils.r")
source("sarimax/src/model_sel.r")  # defines run_model_selection(); the script's
                                    # own loop will also start running unless you
                                    # source only the function definition block
concluded_cities <- data.frame(city = character(0))
result <- run_model_selection(city = 3549805, concluded_cities = concluded_cities)  # São Paulo city
```

(`run_model_selection()` also accepts a `state`/`concluded_states` pair for the legacy state-level path against `processed_data/<disease>/<disease>_<UF>_agg.csv.gz`, but that path isn't exercised by this repository's own scripts.)

## Credentials for `sub_pred.r`

`sub_pred.r` never hardcodes the Mosqlimate API key. Create a `.env` file in the **repository root** (already excluded from Git via `.gitignore`) with:

```
MOSQLIMATE_API_KEY=your-key-here
```

Get your key from the "Auth" section of your Mosqlimate profile at [mosqlimate.org](https://mosqlimate.org/). Also update the `repository` and `commit` constants at the top of `sub_pred.r` to match the GitHub repository and commit hash you are actually submitting.

## Modeling summary

- Response: `log1p(cases)`, back-transformed with `expm1()` and clipped at zero.
- Covariates: PCA components of de-correlated weather variables (contemporaneous, lagged, and rolling-mean), plus ENSO/IOD/PDO when they add signal beyond the retained weather components — see the root README's "Data and Variables" section for the full selection procedure.
- SARIMAX order: grid-searched per city, bounded by `max_order` (default `p≤2, d≤1, q≤2, P≤1, D=0, Q≤1`), fit with `method = "CSS-ML"`, `optim.method = "BFGS"`.
- Selection metric: mean Weighted Interval Score (WIS) across 4 cross-validation splits (configurable via the `metric` argument of `run_model_selection()`).
- Prediction intervals: simulated (`bootstrap = TRUE`, `npaths = 1000`) at 50/80/90/95% coverage.
- Robustness: hard `Arima()` failures, convergence/NaN-SE warnings, and degenerate all-zero point forecasts (common for sparse, low-incidence cities) all trigger an automatic retry with the next-best formula/order for that city; unresolved cases and all warnings/errors/degenerate flags are logged rather than silently dropped (see `fit.r`'s row in the Files table above).

See the root `README.md` for the full Team, Data, Model Training, Data Usage Restriction, Predictive Uncertainty, and References sections required by the IMDC submission format.
