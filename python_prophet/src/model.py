"""Prophet model fitting and multi-quantile prediction."""

import logging
import warnings

import numpy as np
import pandas as pd
from prophet import Prophet

logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

QUANTILES = [0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975]

QUANTILE_TO_COL = {
    0.025: "lower_95",
    0.05: "lower_90",
    0.10: "lower_80",
    0.25: "lower_50",
    0.50: "pred",
    0.75: "upper_50",
    0.90: "upper_80",
    0.95: "upper_90",
    0.975: "upper_95",
}

INTERVAL_COLS_ORDERED = [
    "lower_95", "lower_90", "lower_80", "lower_50",
    "pred",
    "upper_50", "upper_80", "upper_90", "upper_95",
]


def _build_model(uncertainty_samples: int = 1000) -> Prophet:
    m = Prophet(
        yearly_seasonality=False,  # we add it manually with more Fourier terms
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        uncertainty_samples=uncertainty_samples,
    )
    # Dengue has a sharp seasonal peak; fourier_order=10 captures it better than default 10
    m.add_seasonality(name="yearly", period=365.25, fourier_order=10)
    return m


def fit_and_forecast(
    train_df: pd.DataFrame,
    target_dates: pd.DatetimeIndex,
    uncertainty_samples: int = 1000,
) -> pd.DataFrame:
    """
    Fit Prophet on log1p-scaled cases and generate forecasts for target_dates.

    Returns a DataFrame with columns:
        date, pred, lower_50, upper_50, lower_80, upper_80,
        lower_90, upper_90, lower_95, upper_95
    All values are back-transformed (case counts, ≥ 0).
    Intervals are guaranteed nested.
    """
    m = _build_model(uncertainty_samples)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(train_df)

    future = pd.DataFrame({"ds": target_dates})
    samples = m.predictive_samples(future)["yhat"]  # shape: (n_dates, n_samples)

    quantile_vals = np.quantile(samples, QUANTILES, axis=1).T  # (n_dates, n_quantiles)
    # Back-transform from log1p space and clamp
    quantile_vals = np.expm1(quantile_vals).clip(min=0)

    # Build output DataFrame
    result = pd.DataFrame(quantile_vals, columns=[QUANTILE_TO_COL[q] for q in QUANTILES])
    result.insert(0, "date", target_dates)

    # Enforce nested intervals per row
    interval_arr = result[INTERVAL_COLS_ORDERED].values
    interval_arr = np.sort(interval_arr, axis=1)
    result[INTERVAL_COLS_ORDERED] = interval_arr

    # Reorder columns to canonical order
    col_order = ["date"] + INTERVAL_COLS_ORDERED[:4] + ["pred"] + INTERVAL_COLS_ORDERED[5:]
    # Actually use: date, lower_95...lower_50, pred, upper_50...upper_95
    result = result[["date", "lower_95", "lower_90", "lower_80", "lower_50",
                      "pred", "upper_50", "upper_80", "upper_90", "upper_95"]]

    return result


def validate_output(df: pd.DataFrame, label: str = "") -> bool:
    """Check submission rules. Returns True if valid, prints warnings otherwise."""
    ok = True

    if not (df["date"].dt.day_of_week == 6).all():
        print(f"[WARN] {label}: not all dates are Sundays")
        ok = False

    date_diffs = df["date"].sort_values().diff().dropna()
    if not (date_diffs == pd.Timedelta("7D")).all():
        print(f"[WARN] {label}: date sequence has gaps")
        ok = False

    num_cols = ["lower_95", "lower_90", "lower_80", "lower_50",
                "pred", "upper_50", "upper_80", "upper_90", "upper_95"]
    if (df[num_cols] < 0).any().any():
        print(f"[WARN] {label}: negative values detected")
        ok = False

    pairs = [
        ("lower_95", "lower_90"), ("lower_90", "lower_80"), ("lower_80", "lower_50"),
        ("lower_50", "pred"), ("pred", "upper_50"), ("upper_50", "upper_80"),
        ("upper_80", "upper_90"), ("upper_90", "upper_95"),
    ]
    for lo, hi in pairs:
        if (df[lo] > df[hi]).any():
            print(f"[WARN] {label}: interval not nested: {lo} > {hi}")
            ok = False

    return ok
