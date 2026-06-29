# LightGBM Quantile Benchmark Model

This module follows the SARIMAX repository layout and writes benchmark outputs to
`lgbm_quantile/results/`.

Run from the repository root:

```bash
python lgbm_quantile/src/run_states.py
```

Useful overrides:

```bash
DISEASES=dengue STATES=MG,SP N_ESTIMATORS=200 python lgbm_quantile/src/run_states.py
DISEASES=dengue,chikungunya STATES=all python lgbm_quantile/src/run_states.py
```

Outputs:

- `lgbm_quantile/results/pred/predictions_<disease>.csv.gz`
- `lgbm_quantile/results/pred/benchmark_predictions_<disease>.csv.gz`
- `lgbm_quantile/results/metrics/metrics_<disease>.csv`
- `lgbm_quantile/results/metrics/model_selection_<disease>.csv`
- `lgbm_quantile/results/metrics/optimized_ensemble_weights_<disease>.csv`
- `lgbm_quantile/results/metrics/model_comparison.csv`
- `lgbm_quantile/results/metrics/benchmark_summary.csv`
