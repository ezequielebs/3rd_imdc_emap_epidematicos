"""Load and split processed dengue data for each state and validation fold."""

from pathlib import Path

import numpy as np
import pandas as pd

STATES = [
    "AC", "AL", "AM", "AP", "BA", "CE", "DF", "GO", "MA", "MG",
    "MS", "MT", "PA", "PB", "PE", "PI", "PR", "RJ", "RN", "RO",
    "RR", "RS", "SC", "SE", "SP", "TO",
]

# First Sunday of each validation target season (EW41 of cutoff year)
TARGET_START_DATES = {
    1: "2022-10-09",  # EW41 2022
    2: "2023-10-08",  # EW41 2023
    3: "2024-10-06",  # EW41 2024
    4: "2025-10-05",  # EW41 2025
}
N_TARGET_WEEKS = 52  # EW41 → EW40 of following year

REGRESSORS = [
    "temp_med_mean_lag4",
    "precip_med_mean_lag4",
    "enso",
    "pdo",
    "log_cases_lag52",   # same epiweek last year (log1p-scaled)
]


def _add_log_cases_lag52(df: pd.DataFrame) -> pd.DataFrame:
    """Add log1p(cases) shifted by 52 weeks (same-epiweek last year)."""
    df = df.copy()
    df["log_cases"] = np.log1p(df["cases"].clip(lower=0))
    df["log_cases_lag52"] = df["log_cases"].shift(52)
    df = df.drop(columns=["log_cases"])
    return df


def load_state(state: str, data_dir: Path, disease: str = "dengue") -> pd.DataFrame:
    path = data_dir / f"{disease}_{state}_agg.csv.gz"
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = _add_log_cases_lag52(df)
    return df


def get_target_dates(fold: int) -> pd.DatetimeIndex:
    """Return the 52 weekly Sunday dates for a validation fold."""
    start = pd.Timestamp(TARGET_START_DATES[fold])
    return pd.date_range(start=start, periods=N_TARGET_WEEKS, freq="7D")


def _fill_missing_regressors(
    df_full: pd.DataFrame,
    target_df: pd.DataFrame,
    train_df: pd.DataFrame,
    regressors: list[str],
) -> pd.DataFrame:
    """
    For target dates not in df_full, fill regressor values using the
    seasonal (week-of-year) mean from the training period.
    """
    # Merge known values from the full dataset
    avail = df_full[["date"] + regressors].rename(columns={"date": "ds"})
    target_df = target_df.merge(avail, on="ds", how="left")

    missing_mask = target_df[regressors].isna().any(axis=1)
    if not missing_mask.any():
        return target_df

    # Compute seasonal means from training data (no data leakage)
    train_with_week = train_df[["ds"] + regressors].copy()
    train_with_week["week"] = train_with_week["ds"].dt.isocalendar().week.astype(int)
    seasonal_means = train_with_week.groupby("week")[regressors].mean()

    target_df["_week"] = target_df["ds"].dt.isocalendar().week.astype(int)
    for col in regressors:
        mask = target_df[col].isna()
        if mask.any():
            target_df.loc[mask, col] = target_df.loc[mask, "_week"].map(
                seasonal_means[col]
            )
    target_df = target_df.drop(columns=["_week"])
    return target_df


def get_fold_data(
    df: pd.DataFrame,
    fold: int,
    regressors: list[str] = REGRESSORS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (train_df, target_df) both with columns [ds, y/cases] + regressors.

    train_df: log1p-transformed cases, used for model fitting.
    target_df: regressor values for the 52 target Sundays (no cases column).
    """
    train_mask = df[f"train_{fold}"].astype(bool)
    train_df = df[train_mask][["date"] + regressors + ["cases"]].copy()
    train_df = train_df.rename(columns={"date": "ds", "cases": "y"})
    train_df["y"] = np.log1p(train_df["y"].clip(lower=0))
    # Drop rows where lag52 is NaN (first year of data has no lag52)
    train_df = train_df.dropna(subset=["log_cases_lag52"] if "log_cases_lag52" in regressors else []).reset_index(drop=True)

    target_dates = get_target_dates(fold)
    target_df = pd.DataFrame({"ds": target_dates})
    target_df = _fill_missing_regressors(df, target_df, train_df, regressors)

    # Prophet's add_regressor(standardize=True) handles standardization internally
    # using training-set statistics — no manual standardization needed here.
    return train_df, target_df


def get_actual_cases(df: pd.DataFrame, fold: int, target_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Return actual case counts for target dates where data is available."""
    target_mask = df[f"target_{fold}"].astype(bool)
    actual = df[target_mask][["date", "cases"]].copy()
    actual["date"] = pd.to_datetime(actual["date"])
    return actual
