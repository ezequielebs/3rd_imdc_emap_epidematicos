"""
Run Prophet validation forecasts for all states and folds.

Usage:
    uv run python src/run_all.py                   # all 26 states
    STATES=MG,SP uv run python src/run_all.py      # subset
    STATES=MG,SP N_JOBS=2 uv run python src/run_all.py
    RESET_CHECKPOINT=true uv run python src/run_all.py
"""

import logging
import os
import sys
import time
import traceback
from pathlib import Path

# Suppress Stan/Prophet verbose output before any imports that trigger them
os.environ.setdefault("CMDSTAN_VERBOSITY", "0")
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

import joblib
import pandas as pd

# Paths relative to this file's location
SRC_DIR = Path(__file__).parent
ROOT_DIR = SRC_DIR.parent.parent
DATA_DIR = ROOT_DIR / "processed_data" / "dengue"
RESULTS_DIR = SRC_DIR.parent / "results"
PRED_DIR = RESULTS_DIR / "predictions"
METRICS_DIR = RESULTS_DIR / "metrics"
CHECKPOINT_FILE = RESULTS_DIR / "concluded_states.csv"

sys.path.insert(0, str(SRC_DIR))
from data_loader import STATES, get_fold_data, load_state
from model import fit_and_forecast, validate_output


def _wis(pred: pd.Series, lo: pd.Series, hi: pd.Series, alpha: float) -> float:
    """Weighted Interval Score for one interval level."""
    width = (hi - lo).clip(lower=0)
    penalty_lo = (2 / alpha) * (lo - pred).clip(lower=0)
    penalty_hi = (2 / alpha) * (pred - hi).clip(lower=0)
    return float((width + penalty_lo + penalty_hi).mean())


def compute_metrics(pred_df: pd.DataFrame, actual: pd.Series) -> dict:
    """Compute WIS and coverage metrics against actual case counts."""
    p = pred_df["pred"]
    mae = float((p - actual).abs().mean())
    rmse = float(((p - actual) ** 2).mean() ** 0.5)

    intervals = {
        50: ("lower_50", "upper_50", 0.50),
        80: ("lower_80", "upper_80", 0.20),
        90: ("lower_90", "upper_90", 0.10),
        95: ("lower_95", "upper_95", 0.05),
    }

    wis_parts = []
    coverage = {}
    for pct, (lo_col, hi_col, alpha) in intervals.items():
        lo, hi = pred_df[lo_col], pred_df[hi_col]
        wis_parts.append(_wis(actual, lo, hi, alpha))
        coverage[f"coverage_{pct}"] = float(((actual >= lo) & (actual <= hi)).mean())

    # WIS = mean of interval scores weighted equally (simplified)
    wis = float(sum(wis_parts) / len(wis_parts))

    return {"wis": wis, "mae": mae, "rmse": rmse, **coverage}


def process_state(state: str) -> dict:
    """Fit all 4 validation folds for one state. Returns summary dict."""
    t0 = time.time()
    try:
        df = load_state(state, DATA_DIR)
    except FileNotFoundError:
        print(f"[SKIP] {state}: data file not found")
        return {"state": state, "status": "skipped"}

    all_preds = []
    fold_metrics = []

    for fold in range(1, 5):
        train_df, target_dates = get_fold_data(df, fold)
        pred_df = fit_and_forecast(train_df, target_dates)
        pred_df["state"] = state
        pred_df["fold"] = fold
        all_preds.append(pred_df)

        # Compute metrics against actual cases where available in data
        actual_rows = df[df[f"target_{fold}"].astype(bool)][["date", "cases"]].copy()
        actual_rows["date"] = pd.to_datetime(actual_rows["date"])
        merged = pred_df.merge(actual_rows, on="date", how="inner")
        if len(merged) > 0:
            metrics = compute_metrics(merged, merged["cases"])
            metrics["state"] = state
            metrics["fold"] = fold
            metrics["n_obs"] = len(merged)
            fold_metrics.append(metrics)

    # Save predictions
    val_df = pd.concat(all_preds, ignore_index=True)
    out_path = PRED_DIR / f"{state}_validation.csv"
    val_df.to_csv(out_path, index=False)

    # Validate output for fold 1 as a spot check
    fold1 = val_df[val_df["fold"] == 1].copy()
    fold1["date"] = pd.to_datetime(fold1["date"])
    validate_output(fold1, label=f"{state}/fold1")

    # Save metrics
    if fold_metrics:
        metrics_df = pd.DataFrame(fold_metrics)
        metrics_df.to_csv(METRICS_DIR / f"{state}_metrics.csv", index=False)

    elapsed = time.time() - t0
    avg_wis = sum(m["wis"] for m in fold_metrics) / len(fold_metrics) if fold_metrics else float("nan")
    print(f"[OK] {state}: avg WIS={avg_wis:.1f}, elapsed={elapsed:.1f}s")
    return {"state": state, "status": "ok", "wis": avg_wis, "elapsed": elapsed}


def load_checkpoint() -> set[str]:
    if CHECKPOINT_FILE.exists():
        df = pd.read_csv(CHECKPOINT_FILE)
        return set(df["state"].tolist())
    return set()


def save_checkpoint(done: set[str]) -> None:
    pd.DataFrame({"state": sorted(done)}).to_csv(CHECKPOINT_FILE, index=False)


def main() -> None:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    reset = os.environ.get("RESET_CHECKPOINT", "false").lower() == "true"
    if reset and CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        print("Checkpoint reset.")

    states_env = os.environ.get("STATES", "all")
    if states_env.lower() == "all":
        states_to_run = STATES
    else:
        states_to_run = [s.strip().upper() for s in states_env.split(",")]

    done = load_checkpoint()
    pending = [s for s in states_to_run if s not in done]

    if not pending:
        print("All states already concluded. Use RESET_CHECKPOINT=true to rerun.")
        return

    print(f"Running {len(pending)} states: {pending}")
    n_jobs = int(os.environ.get("N_JOBS", "-1"))

    results = joblib.Parallel(n_jobs=n_jobs, prefer="threads", verbose=0)(
        joblib.delayed(_run_state_safe)(state) for state in pending
    )

    for r in results:
        if r and r.get("status") == "ok":
            done.add(r["state"])

    save_checkpoint(done)

    ok = [r for r in results if r and r.get("status") == "ok"]
    failed = [r for r in results if r and r.get("status") not in ("ok", "skipped")]

    print(f"\nDone: {len(ok)} succeeded, {len(failed)} failed, {len(done)} total concluded.")

    if ok:
        wis_values = [r["wis"] for r in ok if "wis" in r and not pd.isna(r["wis"])]
        if wis_values:
            print(f"\nWIS by state (avg across folds):")
            sorted_ok = sorted(ok, key=lambda r: r.get("wis", float("inf")))
            for r in sorted_ok:
                wis = r.get("wis", float("nan"))
                print(f"  {r['state']}: {wis:.1f}")
            print(f"\nOverall average WIS: {sum(wis_values)/len(wis_values):.1f}")

    if failed:
        print("Failed states:", [r["state"] for r in failed])


def _run_state_safe(state: str) -> dict:
    try:
        return process_state(state)
    except Exception:
        print(f"[ERROR] {state}:\n{traceback.format_exc()}")
        return {"state": state, "status": "error"}


if __name__ == "__main__":
    main()
