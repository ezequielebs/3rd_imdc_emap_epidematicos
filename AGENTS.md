# Repository Guidelines

## Session Rules for Agents

Follow the user's workflow strictly. New contributor code should be simple and direct: read fixed inputs, fit or load a model, and write benchmark-compatible predictions. Do not add unnecessary abstractions, generic validators, or production-oriented structure. Do not add unit tests unless explicitly asked. Do not check input types; schemas and paths are fixed.

## Project Structure & Data Locations

Raw source data is in `raw_data/`; do not modify it casually. Prepared modeling data is in `processed_data/`:

- `processed_data/dengue/`: state-level dengue files such as `dengue_SP_agg.csv.gz` and `dengue_merged.csv.gz`.
- `processed_data/chikungunya/`: analogous chikungunya files.
- `processed_data/climate/`: processed climate covariates.

Existing workflows remain in `data_prep/`, `eda/`, `glmm/`, and model folders such as `sarimax/`. New models must follow the SARIMAX layout: `<model_name>/src/` for code, `<model_name>/results/` for outputs, and optional runner scripts at `<model_name>/`.

## Prediction Outputs

Write each model's outputs inside its own results folder. Use `<model_name>/results/metrics/` for metrics and `<model_name>/results/pred/` for prediction CSVs. Use clear filenames with disease, geography, and purpose, for example `baseline/results/pred/dengue_state_predictions.csv`.

Existing SARIMAX outputs stay in `sarimax/results/`. New models should use equivalent paths so they can enter the same benchmark later.

## Development Commands

Run scripts from the repository root so relative paths work.

- `python <model_name>/src/run_states.py`: expected pattern for state-level Python benchmark models.
- `Rscript data_prep/merge_dengue.r`: rebuilds merged dengue data when R preprocessing is needed.
- `Rscript sarimax/run_mg_sp_test.R`: smoke-runs the existing SARIMAX workflow.
- `STATES=MG,SP NPATHS=300 Rscript sarimax/src/run_states.R`: runs SARIMAX for selected states.

## Python Environment

Agents may use the local Python installation to run code. If packages are missing or isolation is useful, create a repository-local virtual environment with `python -m venv .venv`, activate it, and install required packages there. Keep dependency setup simple and document any important install command in the model folder or PR notes.

## Coding Style

Use Python packages when they make modeling, data handling, or evaluation easier. Keep control flow explicit: load data, prepare features, fit model, generate predictions, write CSV. Use readable functions when they shorten or clarify the workflow. Use snake_case for files, variables, and functions. Constants at the top of a script are enough for configuration.

## Validation Expectations

Unit tests are not required for this session. Make scripts runnable end to end on fixed repository data. If checking behavior, run the target script once and confirm it writes files under `<model_name>/results/metrics/` and `<model_name>/results/pred/`. Keep error handling minimal.

## Commit & Pull Request Guidelines

Use short, imperative commit messages such as `Add baseline benchmark model` or `Update SARIMAX workflow`. Pull requests should state which data files were read, which model script was run, and where metrics and predictions were written.
