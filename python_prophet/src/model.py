"""Prophet model fitting with block-bootstrap intervals and multi-quantile output."""

import logging
import warnings

import numpy as np
import pandas as pd
from prophet import Prophet

logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

QUANTILES = [0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975]

QUANTILE_TO_COL = {
    0.025: "lower_95",
    0.05:  "lower_90",
    0.10:  "lower_80",
    0.25:  "lower_50",
    0.50:  "pred",
    0.75:  "upper_50",
    0.90:  "upper_80",
    0.95:  "upper_90",
    0.975: "upper_95",
}

OUTPUT_COLS = [
    "date", "lower_95", "lower_90", "lower_80", "lower_50",
    "pred", "upper_50", "upper_80", "upper_90", "upper_95",
]

# Expanded regressors — mirrors SARIMAX's multi-variable approach
REGRESSORS = [
    "temp_med_mean_lag4",
    "precip_med_mean_lag4",
    "enso",
    "pdo",
    "log_cases_lag52",   # same epiweek last year (log1p-scaled)
]


def _build_model(regressors: list[str]) -> Prophet:
    m = Prophet(
        growth="flat",              # dengue is seasonal, not trending; prevents extrapolation
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="additive",  # additive on log scale = multiplicative in original space
        uncertainty_samples=0,
    )
    m.add_seasonality(name="yearly", period=365.25, fourier_order=10)
    for reg in regressors:
        m.add_regressor(reg, standardize=True)
    return m


def _seasonal_bootstrap(
    residuals: np.ndarray,
    train_dates: pd.DatetimeIndex,
    target_dates: pd.DatetimeIndex,
    n_paths: int,
    epiweek_window: int = 4,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Seasonal-stratified bootstrap: for each target week, sample residuals
    from the same ±epiweek_window window in the training set.

    Captures heteroscedasticity: wide uncertainty at outbreak peaks,
    tight uncertainty off-season. Returns (n_paths, n_target).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    train_weeks = train_dates.isocalendar().week.values.astype(int)
    target_weeks = target_dates.isocalendar().week.values.astype(int)
    n_target = len(target_dates)

    paths = np.zeros((n_paths, n_target))
    for j, week in enumerate(target_weeks):
        nearby = {((week - 1 + dw) % 52) + 1 for dw in range(-epiweek_window, epiweek_window + 1)}
        candidate_idx = np.where(np.isin(train_weeks, list(nearby)))[0]
        if len(candidate_idx) < 5:
            candidate_idx = np.arange(len(residuals))  # fallback: use all
        idx = rng.integers(0, len(candidate_idx), size=n_paths)
        paths[:, j] = residuals[candidate_idx[idx]]

    return paths


def fit_and_forecast(
    train_df: pd.DataFrame,
    target_df: pd.DataFrame,
    regressors: list[str] = REGRESSORS,
    n_bootstrap: int = 1000,
    epiweek_window: int = 4,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Fit Prophet and generate multi-quantile forecasts via seasonal-stratified bootstrap.

    For each target week, residuals are sampled from the same ±epiweek_window in
    the training set (heteroscedastic: wide at epidemic peak, narrow off-season).

    Parameters
    ----------
    train_df : DataFrame with [ds, y, *regressors]; y is log1p(cases).
    target_df : DataFrame with [ds, *regressors] for forecast dates.
    regressors : covariate column names.
    n_bootstrap : bootstrap paths.
    epiweek_window : ± weeks around each epiweek to sample residuals from.
    seed : random seed.

    Returns
    -------
    DataFrame with OUTPUT_COLS, back-transformed to case counts (≥ 0).
    """
    rng = np.random.default_rng(seed)
    fit_cols = ["ds", "y"] + regressors
    m = _build_model(regressors)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(train_df[fit_cols])

    train_pred = m.predict(train_df[fit_cols])
    residuals = (train_df["y"].values - train_pred["yhat"].values).astype(float)
    train_dates = pd.DatetimeIndex(train_df["ds"])

    forecast = m.predict(target_df[["ds"] + regressors])
    yhat_log = forecast["yhat"].values
    target_dates = pd.DatetimeIndex(target_df["ds"])

    boot_residuals = _seasonal_bootstrap(
        residuals, train_dates, target_dates, n_bootstrap, epiweek_window, rng
    )
    bootstrap_log = yhat_log[np.newaxis, :] + boot_residuals

    quantile_log = np.quantile(bootstrap_log, QUANTILES, axis=0).T
    quantile_vals = np.expm1(quantile_log).clip(min=0)

    result = pd.DataFrame(quantile_vals, columns=[QUANTILE_TO_COL[q] for q in QUANTILES])
    result.insert(0, "date", target_df["ds"].values)

    interval_cols = [c for c in OUTPUT_COLS if c != "date"]
    result[interval_cols] = np.sort(result[interval_cols].values, axis=1)

    return result[OUTPUT_COLS]


# ── WIS (Bracher et al. 2021, matching R scoringutils) ───────────────────────

def _interval_score(actual: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float) -> float:
    width = hi - lo
    pen_lo = (2.0 / alpha) * np.maximum(lo - actual, 0)
    pen_hi = (2.0 / alpha) * np.maximum(actual - hi, 0)
    return float(np.mean(width + pen_lo + pen_hi))


def wis(
    actual: np.ndarray,
    pred: np.ndarray,
    lower_50: np.ndarray, upper_50: np.ndarray,
    lower_80: np.ndarray, upper_80: np.ndarray,
    lower_90: np.ndarray, upper_90: np.ndarray,
    lower_95: np.ndarray, upper_95: np.ndarray,
) -> float:
    """
    Weighted Interval Score per Bracher et al. 2021 (matches scoringutils).

    WIS = 1/(K + 0.5) * [0.5*|y - m| + Σ_k (alpha_k/2) * IS_{alpha_k}]
    K=4 intervals → denominator = 4.5
    """
    intervals = [
        (lower_50, upper_50, 0.50, 0.25),
        (lower_80, upper_80, 0.20, 0.10),
        (lower_90, upper_90, 0.10, 0.05),
        (lower_95, upper_95, 0.05, 0.025),
    ]
    weighted = 0.5 * float(np.mean(np.abs(actual - pred)))
    for lo, hi, alpha, weight in intervals:
        weighted += weight * _interval_score(actual, lo, hi, alpha)
    return weighted / 4.5


def compute_metrics(pred_df: pd.DataFrame, actual: np.ndarray) -> dict:
    p = pred_df["pred"].values
    w = wis(
        actual, p,
        pred_df["lower_50"].values, pred_df["upper_50"].values,
        pred_df["lower_80"].values, pred_df["upper_80"].values,
        pred_df["lower_90"].values, pred_df["upper_90"].values,
        pred_df["lower_95"].values, pred_df["upper_95"].values,
    )
    coverage = {}
    for pct, lo_col, hi_col in [
        (50, "lower_50", "upper_50"), (80, "lower_80", "upper_80"),
        (90, "lower_90", "upper_90"), (95, "lower_95", "upper_95"),
    ]:
        lo, hi = pred_df[lo_col].values, pred_df[hi_col].values
        coverage[f"coverage_{pct}"] = float(np.mean((actual >= lo) & (actual <= hi)))
    return {
        "wis": w,
        "mae": float(np.mean(np.abs(p - actual))),
        "rmse": float(np.sqrt(np.mean((p - actual) ** 2))),
        **coverage,
    }


def validate_output(df: pd.DataFrame, label: str = "") -> bool:
    ok = True
    if not (pd.to_datetime(df["date"]).dt.day_of_week == 6).all():
        print(f"[WARN] {label}: not all dates are Sundays")
        ok = False
    diffs = pd.to_datetime(df["date"]).sort_values().diff().dropna()
    if not (diffs == pd.Timedelta("7D")).all():
        print(f"[WARN] {label}: date gaps detected")
        ok = False
    num_cols = [c for c in OUTPUT_COLS if c != "date"]
    if (df[num_cols] < 0).any().any():
        print(f"[WARN] {label}: negative values")
        ok = False
    return ok
