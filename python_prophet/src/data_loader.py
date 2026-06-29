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
# Derived from actual data: target_N column start dates
TARGET_START_DATES = {
    1: "2022-10-09",  # EW41 2022
    2: "2023-10-08",  # EW41 2023
    3: "2024-10-06",  # EW41 2024
    4: "2025-10-05",  # EW41 2025
}
N_TARGET_WEEKS = 52  # EW41 → EW40 of following year


def load_state(state: str, data_dir: Path) -> pd.DataFrame:
    path = data_dir / f"dengue_{state}_agg.csv.gz"
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def get_target_dates(fold: int) -> pd.DatetimeIndex:
    """Return the 52 weekly Sunday dates for a validation fold."""
    start = pd.Timestamp(TARGET_START_DATES[fold])
    return pd.date_range(start=start, periods=N_TARGET_WEEKS, freq="7D")


def get_fold_data(df: pd.DataFrame, fold: int) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """
    Returns (train_df, target_dates).

    train_df has columns ds and y (log1p-transformed cases).
    target_dates is a DatetimeIndex covering all 52 target weeks.
    """
    train_mask = df[f"train_{fold}"].astype(bool)
    train_df = df[train_mask][["date", "cases"]].copy()
    train_df = train_df.rename(columns={"date": "ds", "cases": "y"})
    train_df["y"] = np.log1p(train_df["y"].clip(lower=0))

    target_dates = get_target_dates(fold)
    return train_df, target_dates


def get_forecast_train(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return training data for the final forecast target.
    Uses all available data (most recent update covers up to EW25 of the target year).
    """
    train_df = df[["date", "cases"]].copy()
    train_df = train_df.rename(columns={"date": "ds", "cases": "y"})
    train_df["y"] = np.log1p(train_df["y"].clip(lower=0))
    return train_df
