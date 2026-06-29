"""
Run Prophet validation forecasts for all states and folds.

Usage:
    uv run python src/run_all.py                    # all 26 states
    STATES=MG,SP uv run python src/run_all.py       # subset
    RESET_CHECKPOINT=true uv run python src/run_all.py
"""

import logging
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("CMDSTAN_VERBOSITY", "0")
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

import joblib
import pandas as pd

SRC_DIR = Path(__file__).parent
ROOT_DIR = SRC_DIR.parent.parent
DISEASE = os.environ.get("DISEASE", "dengue")
DATA_DIR = ROOT_DIR / "processed_data" / DISEASE
RESULTS_DIR = SRC_DIR.parent / "results" / DISEASE
PRED_DIR = RESULTS_DIR / "predictions"
METRICS_DIR = RESULTS_DIR / "metrics"
CHECKPOINT_FILE = RESULTS_DIR / "concluded_states.csv"

sys.path.insert(0, str(SRC_DIR))
from data_loader import STATES, REGRESSORS, get_actual_cases, get_fold_data, load_state
from model import compute_metrics, fit_and_forecast, validate_output


def process_state(state: str) -> dict:
    t0 = time.time()
    try:
        df = load_state(state, DATA_DIR, DISEASE)
    except FileNotFoundError:
        print(f"[SKIP] {state}: data file not found")
        return {"state": state, "status": "skipped"}

    all_preds = []
    fold_metrics = []

    for fold in range(1, 5):
        train_df, target_df = get_fold_data(df, fold, regressors=REGRESSORS)
        pred_df = fit_and_forecast(train_df, target_df, regressors=REGRESSORS)
        pred_df["state"] = state
        pred_df["fold"] = fold
        all_preds.append(pred_df)

        actual = get_actual_cases(df, fold, target_df["ds"])
        merged = pred_df.merge(actual, on="date", how="inner")
        if len(merged) > 0:
            metrics = compute_metrics(merged, merged["cases"].values)
            metrics.update({"state": state, "fold": fold, "n_obs": len(merged)})
            fold_metrics.append(metrics)

    val_df = pd.concat(all_preds, ignore_index=True)
    val_df.to_csv(PRED_DIR / f"{state}_validation.csv", index=False)

    fold1 = val_df[val_df["fold"] == 1].copy()
    validate_output(fold1, label=f"{state}/fold1")

    if fold_metrics:
        pd.DataFrame(fold_metrics).to_csv(METRICS_DIR / f"{state}_metrics.csv", index=False)

    elapsed = time.time() - t0
    avg_wis = sum(m["wis"] for m in fold_metrics) / len(fold_metrics) if fold_metrics else float("nan")
    print(f"[OK] {state}: avg WIS={avg_wis:.2f}, elapsed={elapsed:.1f}s")
    return {"state": state, "status": "ok", "wis": avg_wis, "elapsed": elapsed}


def _run_state_safe(state: str) -> dict:
    try:
        return process_state(state)
    except Exception:
        print(f"[ERROR] {state}:\n{traceback.format_exc()}")
        return {"state": state, "status": "error"}


def load_checkpoint() -> set[str]:
    if CHECKPOINT_FILE.exists():
        return set(pd.read_csv(CHECKPOINT_FILE)["state"].tolist())
    return set()


def save_checkpoint(done: set[str]) -> None:
    pd.DataFrame({"state": sorted(done)}).to_csv(CHECKPOINT_FILE, index=False)


def main() -> None:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    if os.environ.get("RESET_CHECKPOINT", "false").lower() == "true" and CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        print("Checkpoint reset.")

    states_env = os.environ.get("STATES", "all")
    states_to_run = STATES if states_env.lower() == "all" else [
        s.strip().upper() for s in states_env.split(",")
    ]

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
        wis_values = [r["wis"] for r in ok if not pd.isna(r.get("wis", float("nan")))]
        if wis_values:
            sorted_ok = sorted(ok, key=lambda r: r.get("wis", float("inf")))
            print("\nWIS by state (avg across folds):")
            for r in sorted_ok:
                marker = " ✓" if r.get("wis", float("inf")) < 1072.95 else ""
                print(f"  {r['state']}: {r.get('wis', float('nan')):.2f}{marker}")
            avg = sum(wis_values) / len(wis_values)
            target = 1072.95
            status = "✓ BEATS TARGET" if avg < target else f"✗ target={target:.2f}"
            print(f"\nOverall average WIS: {avg:.2f}  [{status}]")

    if failed:
        print("Failed states:", [r["state"] for r in failed])


if __name__ == "__main__":
    main()
