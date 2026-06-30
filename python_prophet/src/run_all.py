"""
Run Prophet validation forecasts for all states/cities and folds.

Usage:
    uv run python src/run_all.py                              # all 26 states (dengue)
    uv run python src/run_all.py --states MG,SP               # state subset
    uv run python src/run_all.py --disease chikungunya        # different disease
    uv run python src/run_all.py --scope city                 # city-level forecasts
    uv run python src/run_all.py --scope city --disease chikungunya
    uv run python src/run_all.py --reset-checkpoint           # clear checkpoint
    uv run python src/run_all.py --n-jobs 4                   # parallelism
    # env vars still work as fallback:
    STATES=MG,SP uv run python src/run_all.py
    RESET_CHECKPOINT=true uv run python src/run_all.py

Output: one CSV per (unit, fold) — e.g. AC_target_1.csv, 2931350_target_1.csv
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
from data_loader import (
    STATES, REGRESSORS, CITY_REGRESSORS,
    DENGUE_CITIES, CHIKUNGUNYA_CITIES,
    get_actual_cases, get_fold_data, load_state, load_city,
)
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
    parser.add_argument(
        "--scope",
        default=None,
        choices=["state", "city"],
        help="Forecast scope: 'state' (default) or 'city'. Overrides SCOPE env var.",
    )
    return parser.parse_args()


def _resolve(args: argparse.Namespace) -> tuple[str, str, list[str], bool, int]:
    disease = args.disease or os.environ.get("DISEASE", "dengue")
    scope = args.scope or os.environ.get("SCOPE", "state")

    states_arg = args.states or os.environ.get("STATES", "all")
    states_to_run = STATES if states_arg.lower() == "all" else [
        s.strip().upper() for s in states_arg.split(",")
    ]

    reset = args.reset_checkpoint or os.environ.get("RESET_CHECKPOINT", "false").lower() == "true"

    n_jobs = args.n_jobs if args.n_jobs is not None else int(os.environ.get("N_JOBS", "-1"))

    return disease, scope, states_to_run, reset, n_jobs


def _process_unit(
    unit_id: str,
    df: pd.DataFrame,
    regressors: list[str],
    pred_dir: Path,
    metrics_dir: Path,
) -> dict:
    t0 = time.time()
    fold_metrics = []

    for fold in range(1, 5):
        train_df, target_df = get_fold_data(df, fold, regressors=regressors)
        pred_df = fit_and_forecast(train_df, target_df, regressors=regressors)
        pred_df["unit"] = unit_id
        pred_df["fold"] = fold

        pred_df.to_csv(pred_dir / f"{unit_id}_target_{fold}.csv", index=False)
        validate_output(pred_df, label=f"{unit_id}/fold{fold}")

        actual = get_actual_cases(df, fold, target_df["ds"])
        merged = pred_df.merge(actual, on="date", how="inner")
        if len(merged) > 0:
            metrics = compute_metrics(merged, merged["cases"].values)
            metrics.update({"unit": unit_id, "fold": fold, "n_obs": len(merged)})
            fold_metrics.append(metrics)

    if fold_metrics:
        pd.DataFrame(fold_metrics).to_csv(metrics_dir / f"{unit_id}_metrics.csv", index=False)

    elapsed = time.time() - t0
    avg_wis = sum(m["wis"] for m in fold_metrics) / len(fold_metrics) if fold_metrics else float("nan")
    print(f"[OK] {unit_id}: avg WIS={avg_wis:.2f}, elapsed={elapsed:.1f}s")
    return {"unit": unit_id, "status": "ok", "wis": avg_wis, "elapsed": elapsed}


def process_state(state: str, data_dir: Path, disease: str, pred_dir: Path, metrics_dir: Path) -> dict:
    try:
        df = load_state(state, data_dir, disease)
    except FileNotFoundError:
        print(f"[SKIP] {state}: data file not found")
        return {"unit": state, "status": "skipped"}
    return _process_unit(state, df, REGRESSORS, pred_dir, metrics_dir)


def process_city(geocode: int, data_dir: Path, disease: str, pred_dir: Path, metrics_dir: Path) -> dict:
    try:
        df = load_city(geocode, data_dir, disease)
    except FileNotFoundError:
        print(f"[SKIP] {geocode}: data file not found")
        return {"unit": str(geocode), "status": "skipped"}
    return _process_unit(str(geocode), df, CITY_REGRESSORS, pred_dir, metrics_dir)


def _run_state_safe(state: str, data_dir: Path, disease: str, pred_dir: Path, metrics_dir: Path) -> dict:
    try:
        return process_state(state, data_dir, disease, pred_dir, metrics_dir)
    except Exception:
        print(f"[ERROR] {state}:\n{traceback.format_exc()}")
        return {"unit": state, "status": "error"}


def _run_city_safe(geocode: int, data_dir: Path, disease: str, pred_dir: Path, metrics_dir: Path) -> dict:
    try:
        return process_city(geocode, data_dir, disease, pred_dir, metrics_dir)
    except Exception:
        print(f"[ERROR] {geocode}:\n{traceback.format_exc()}")
        return {"unit": str(geocode), "status": "error"}


def load_checkpoint(checkpoint_file: Path) -> set[str]:
    if checkpoint_file.exists():
        df = pd.read_csv(checkpoint_file)
        col = "unit" if "unit" in df.columns else "state"
        return set(df[col].astype(str).tolist())
    return set()


def save_checkpoint(done: set[str], checkpoint_file: Path) -> None:
    pd.DataFrame({"unit": sorted(done)}).to_csv(checkpoint_file, index=False)


def main() -> None:
    args = parse_args()
    disease, scope, states_to_run, reset, n_jobs = _resolve(args)

    data_dir = ROOT_DIR / "processed_data" / disease
    results_dir = SRC_DIR.parent / "results" / disease / scope
    pred_dir = results_dir / "predictions"
    metrics_dir = results_dir / "metrics"
    checkpoint_file = results_dir / "concluded_units.csv"

    pred_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    if reset and checkpoint_file.exists():
        checkpoint_file.unlink()
        print("Checkpoint reset.")

    done = load_checkpoint(checkpoint_file)

    if scope == "state":
        pending_states = [s for s in states_to_run if s not in done]
        if not pending_states:
            print("All states already concluded. Use --reset-checkpoint to rerun.")
            return
        print(f"Running {len(pending_states)} states ({disease}): {pending_states}")
        results = joblib.Parallel(n_jobs=n_jobs, prefer="threads", verbose=0)(
            joblib.delayed(_run_state_safe)(state, data_dir, disease, pred_dir, metrics_dir)
            for state in pending_states
        )
    else:
        city_map = DENGUE_CITIES if disease == "dengue" else CHIKUNGUNYA_CITIES
        pending_cities = [gc for gc in city_map if str(gc) not in done]
        if not pending_cities:
            print("All cities already concluded. Use --reset-checkpoint to rerun.")
            return
        print(f"Running {len(pending_cities)} cities ({disease}): {pending_cities}")
        results = joblib.Parallel(n_jobs=n_jobs, prefer="threads", verbose=0)(
            joblib.delayed(_run_city_safe)(gc, data_dir, disease, pred_dir, metrics_dir)
            for gc in pending_cities
        )

    for r in results:
        if r and r.get("status") == "ok":
            done.add(r["unit"])
    save_checkpoint(done, checkpoint_file)

    ok = [r for r in results if r and r.get("status") == "ok"]
    failed = [r for r in results if r and r.get("status") not in ("ok", "skipped")]

    print(f"\nDone: {len(ok)} succeeded, {len(failed)} failed, {len(done)} total concluded.")

    if ok:
        wis_values = [r["wis"] for r in ok if not pd.isna(r.get("wis", float("nan")))]
        if wis_values:
            sorted_ok = sorted(ok, key=lambda r: r.get("wis", float("inf")))
            label = "state" if scope == "state" else "city"
            print(f"\nWIS by {label} (avg across folds):")
            for r in sorted_ok:
                marker = " ✓" if r.get("wis", float("inf")) < 1072.95 else ""
                print(f"  {r['unit']}: {r.get('wis', float('nan')):.2f}{marker}")
            avg = sum(wis_values) / len(wis_values)
            target = 1072.95
            status = "✓ BEATS TARGET" if avg < target else f"✗ target={target:.2f}"
            print(f"\nOverall average WIS: {avg:.2f}  [{status}]")

    if failed:
        print("Failed units:", [r["unit"] for r in failed])


if __name__ == "__main__":
    main()
