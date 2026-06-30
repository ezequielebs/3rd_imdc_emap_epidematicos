"""Load and split processed dengue/chikungunya data for states and cities."""

from pathlib import Path

import numpy as np
import pandas as pd
from epiweeks import Week, Year

STATES = [
    "AC", "AL", "AM", "AP", "BA", "CE", "DF", "GO", "MA", "MG",
    "MS", "MT", "PA", "PB", "PE", "PI", "PR", "RJ", "RN", "RO",
    "RR", "RS", "SC", "SE", "SP", "TO",
]

# Dengue Optional Challenge 1 cities (geocode → state)
DENGUE_CITIES: dict[int, str] = {
    2931350: "BA",  # Teixeira de Freitas
    2933307: "BA",  # Vitória da Conquista
    2302503: "CE",  # Brejo Santo
    3119401: "MG",  # Coronel Fabriciano
    3549805: "SP",  # São José do Rio Preto
    3541406: "SP",  # Presidente Prudente
    1200401: "AC",  # Rio Branco
    1200203: "AC",  # Cruzeiro do Sul
    1716109: "TO",  # Paraíso do Tocantins
    4113700: "PR",  # Londrina
    4103701: "PR",  # Cambé
    4104808: "PR",  # Cascavel
    5201405: "GO",  # Aparecida de Goiânia
    5102637: "MT",  # Campo Novo do Parecis
    5215231: "GO",  # Novo Gama
}

# Chikungunya Optional Challenge 3 cities
CHIKUNGUNYA_CITIES: dict[int, str] = {
    2211001: "PI",  # Teresina
    2931350: "BA",  # Teixeira de Freitas
    3143302: "MG",  # Montes Claros
    3119401: "MG",  # Coronel Fabriciano
    1721000: "TO",  # Palmas
    1716109: "TO",  # Paraíso do Tocantins
    4104808: "PR",  # Cascavel
    4219507: "SC",  # Xanxerê
    5103403: "MT",  # Cuiabá
    5102637: "MT",  # Campo Novo do Parecis
}

# First Sunday of each validation target season (EW41 of cutoff year)
TARGET_START_DATES = {
    1: "2022-10-09",  # EW41 2022
    2: "2023-10-08",  # EW41 2023
    3: "2024-10-06",  # EW41 2024
    4: "2025-10-05",  # EW41 2025
}


def _n_target_weeks(start: pd.Timestamp) -> int:
    """
    Number of epiweeks spanning EW41 of `start`'s epiweek-year through EW40 of
    the following year (inclusive). This is normally 52, but some years have
    53 epiweeks (e.g. 2025), which pushes the season to 53 weeks. Hardcoding
    52 silently drops the final week, which the Mosqlimate API then rejects
    as a missing date.
    """
    start_week = Week.fromdate(start.date())
    weeks_left_in_start_year = Year(start_week.year).totalweeks() - start_week.week + 1
    return weeks_left_in_start_year + 40  # ... through EW40 of the next year

REGRESSORS = [
    "temp_med_mean_lag4",
    "precip_med_mean_lag4",
    "enso",
    "pdo",
    "log_cases_lag52",   # same epiweek last year (log1p-scaled)
]

# City files use different column names (no _mean aggregation suffix, no pre-computed lags)
CITY_REGRESSORS = [
    "temp_med_lag4",
    "precip_med_lag4",
    "enso",
    "pdo",
    "log_cases_lag52",
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


def load_city(geocode: int, data_dir: Path, disease: str = "dengue") -> pd.DataFrame:
    """Load city-level data, computing lag4 climate features and log_cases_lag52."""
    path = data_dir / "sel_cities" / f"{disease}_{geocode}.csv.gz"
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    # Normalise case column to match state pipeline
    df = df.rename(columns={"casos": "cases"})
    # Forward-fill sporadic NaN in slowly-varying climate indices
    for col in ["enso", "iod", "pdo", "temp_med", "precip_med"]:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()
    # Compute lag4 climate regressors (not pre-computed in city files)
    df["temp_med_lag4"] = df["temp_med"].shift(4)
    df["precip_med_lag4"] = df["precip_med"].shift(4)
    df = _add_log_cases_lag52(df)
    return df


def get_target_dates(fold: int) -> pd.DatetimeIndex:
    """Return the weekly Sunday dates (EW41 → EW40 of the following year) for a validation fold."""
    start = pd.Timestamp(TARGET_START_DATES[fold])
    n_weeks = _n_target_weeks(start)
    return pd.date_range(start=start, periods=n_weeks, freq="7D")


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
