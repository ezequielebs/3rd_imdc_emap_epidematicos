"""
Run Prophet validation forecasts for all states and folds.

Usage:
    uv run python src/run_all.py                         # all 26 states
    uv run python src/run_all.py --states MG,SP          # subset
    uv run python src/run_all.py --disease chikungunya   # different disease
    uv run python src/run_all.py --reset-checkpoint      # clear checkpoint
    uv run python src/run_all.py --n-jobs 4              # parallelism
    # env vars still work as fallback:
    STATES=MG,SP uv run python src/run_all.py
    RESET_CHECKPOINT=true uv run python src/run_all.py
"""

import argparse
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

sys.path.insert(0, str(SRC_DIR))
from data_loader import STATES, REGRESSORS, get_actual_cases, get_fold_data, load_state
from model import compute_metrics, fit_and_forecast, validate_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Prophet validation forecasts.")
    parser.add_argument(
        "--states",
        default=None,
        help="Comma-separated state codes to run (default: all). Overrides STATES env var.",
    )
    parser.add_argument(
        "--disease",
        default=None,
        choices=["dengue", "chikungunya"],
        help="Disease to forecast (default: dengue). Overrides DISEASE env var.",
    )
    parser.add_argument(
        "--reset-checkpoint",
        action="store_true",
        default=None,
        help="Clear the checkpoint before running. Overrides RESET_CHECKPOINT env var.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Parallel jobs for joblib (default: -1 = all CPUs). Overrides N_JOBS env var.",
    )
    return parser.parse_args()


def _resolve(args: argparse.Namespace) -> tuple[str, list[str], bool, int]:
    disease = args.disease or os.environ.get("DISEASE", "dengue")

    states_arg = args.states or os.environ.get("STATES", "all")
    states_to_run = STATES if states_arg.lower() == "all" else [
        s.strip().upper() for s in states_arg.split(",")
    ]

    reset = args.reset_checkpoint or os.environ.get("RESET_CHECKPOINT", "false").lower() == "true"

    n_jobs = args.n_jobs if args.n_jobs is not None else int(os.environ.get("N_JOBS", "-1"))

    return disease, states_to_run, reset, n_jobs


def process_state(state: str, data_dir: Path, disease: str, pred_dir: Path, metrics_dir: Path) -> dict:
    t0 = time.time()
    try:
        df = load_state(state, data_dir, disease)
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
    val_df.to_csv(pred_dir / f"{state}_validation.csv", index=False)

    fold1 = val_df[val_df["fold"] == 1].copy()
    validate_output(fold1, label=f"{state}/fold1")

    if fold_metrics:
        pd.DataFrame(fold_metrics).to_csv(metrics_dir / f"{state}_metrics.csv", index=False)

    elapsed = time.time() - t0
    avg_wis = sum(m["wis"] for m in fold_metrics) / len(fold_metrics) if fold_metrics else float("nan")
    print(f"[OK] {state}: avg WIS={avg_wis:.2f}, elapsed={elapsed:.1f}s")
    return {"state": state, "status": "ok", "wis": avg_wis, "elapsed": elapsed}


def _run_state_safe(state: str, data_dir: Path, disease: str, pred_dir: Path, metrics_dir: Path) -> dict:
    try:
        return process_state(state, data_dir, disease, pred_dir, metrics_dir)
    except Exception:
        print(f"[ERROR] {state}:\n{traceback.format_exc()}")
        return {"state": state, "status": "error"}


def load_checkpoint(checkpoint_file: Path) -> set[str]:
    if checkpoint_file.exists():
        return set(pd.read_csv(checkpoint_file)["state"].tolist())
    return set()


def save_checkpoint(done: set[str], checkpoint_file: Path) -> None:
    pd.DataFrame({"state": sorted(done)}).to_csv(checkpoint_file, index=False)


def main() -> None:
    args = parse_args()
    disease, states_to_run, reset, n_jobs = _resolve(args)

    data_dir = ROOT_DIR / "processed_data" / disease
    results_dir = SRC_DIR.parent / "results" / disease
    pred_dir = results_dir / "predictions"
    metrics_dir = results_dir / "metrics"
    checkpoint_file = results_dir / "concluded_states.csv"

    pred_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    if reset and checkpoint_file.exists():
        checkpoint_file.unlink()
        print("Checkpoint reset.")

    done = load_checkpoint(checkpoint_file)
    pending = [s for s in states_to_run if s not in done]

    if not pending:
        print("All states already concluded. Use --reset-checkpoint to rerun.")
        return

    print(f"Running {len(pending)} states: {pending}")

    results = joblib.Parallel(n_jobs=n_jobs, prefer="threads", verbose=0)(
        joblib.delayed(_run_state_safe)(state, data_dir, disease, pred_dir, metrics_dir)
        for state in pending
    )

    for r in results:
        if r and r.get("status") == "ok":
            done.add(r["state"])
    save_checkpoint(done, checkpoint_file)

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
