"""
Submit Prophet forecasts to the Mosqlimate Predictions Registry.

Reads the dengue / chikungunya forecasts produced by `run_all.py` (one CSV
per unit x target window under `results/<disease>/<scope>/predictions/`) and
uploads them via the `mosqlient` Python package.

Layout this script expects (as produced by run_all.py):
    results/<disease>/state/predictions/<UF>_target_<fold>.csv      (adm_level=1)
    results/<disease>/city/predictions/<geocode>_target_<fold>.csv  (adm_level=2)

State/city admin codes (adm_1, adm_2) are not stored in the prediction CSVs
themselves, so they are looked up from the matching processed-data file:
    processed_data/<disease>/<disease>_<UF>_agg.csv.gz                 -> uf_code
    processed_data/<disease>/sel_cities/<disease>_<geocode>_agg.csv.gz -> uf_code, geocode

Credentials: the Mosqlimate API key is NEVER hardcoded here. It is read
from the MOSQLIMATE_API_KEY environment variable, which should be defined
in a local, git-ignored `.env` file at the repo root:
    MOSQLIMATE_API_KEY=your-key-here
Get your key from the "Auth" section of your Mosqlimate profile
(https://mosqlimate.org/).

Usage:
    uv run python src/submit_predictions.py --dry-run                          # preview everything
    uv run python src/submit_predictions.py --disease dengue --scope city \
        --repository EzequielEBS/3rd_imdc_emap_epidematicos_prophet \
        --commit <commit_hash>
    uv run python src/submit_predictions.py --disease chikungunya --scope state \
        --repository EzequielEBS/3rd_imdc_emap_epidematicos_prophet \
        --commit <commit_hash> --units AC,SP
"""

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

SRC_DIR = Path(__file__).parent
ROOT_DIR = SRC_DIR.parent.parent

sys.path.insert(0, str(SRC_DIR))
from data_loader import STATES, DENGUE_CITIES, CHIKUNGUNYA_CITIES  # noqa: E402

load_dotenv(ROOT_DIR / ".env")

# ICD-10 disease codes expected by the Mosqlimate API.
DISEASE_CODES = {"dengue": "A90", "chikungunya": "A92.0"}

TARGET_IDS = ["target_1", "target_2", "target_3", "target_4"]


def _default_units(disease: str, scope: str) -> list:
    """Units that actually have forecasts produced by run_all.py."""
    if scope == "state":
        return STATES
    city_map = DENGUE_CITIES if disease == "dengue" else CHIKUNGUNYA_CITIES
    return list(city_map)


def _get_adm_codes(unit, scope: str, disease: str, data_dir: Path) -> tuple:
    """Look up (adm_1, adm_2) for a unit from its processed-data file."""
    if scope == "state":
        path = data_dir / f"{disease}_{unit}_agg.csv.gz"
        df = pd.read_csv(path, usecols=["uf_code"], nrows=1)
        return int(df["uf_code"].iloc[0]), None

    path = data_dir / "sel_cities" / f"{disease}_{unit}_agg.csv.gz"
    df = pd.read_csv(path, usecols=["uf_code", "geocode"], nrows=1)
    return int(df["uf_code"].iloc[0]), int(df["geocode"].iloc[0])


def _upload_one(mosq, max_retries: int = 3, **kwargs) -> tuple:
    """
    Upload a single prediction, retrying on transient (502/503/504) server
    errors with backoff. Returns (outcome, reason):
        ("ok", None)             - uploaded successfully
        ("skip_target", reason)  - skip just this (unit, target) and continue
        ("skip_unit", reason)    - skip remaining targets for this unit
    Any other exception propagates and aborts the run.
    """
    for attempt in range(1, max_retries + 1):
        try:
            mosq.upload_prediction(**kwargs)
            return "ok", None
        except ValueError as e:
            if "Duplication found" in str(e):
                return "skip_unit", "duplicate"
            raise
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status in (502, 503, 504) and attempt < max_retries:
                wait = 5 * attempt
                print(f"[RETRY] server returned {status}; retrying in {wait}s ({attempt}/{max_retries})...")
                time.sleep(wait)
                continue
            return "skip_target", f"http_{status or 'error'}"
    return "skip_target", "retries_exhausted"


def submit_predictions(
    api_key: str,
    repository: str,
    commit: str,
    disease: str,
    scope: str,
    units: list,
    case_definition: str = "probable",
    published: bool = True,
    results_dir: Path = None,
    data_dir: Path = None,
    target_ids: list = TARGET_IDS,
    dry_run: bool = False,
) -> list:
    """
    Upload every (unit x target window) forecast for a disease/scope to
    the Mosqlimate Predictions Registry.

    Parameters
    ----------
    api_key : Mosqlimate API key (read from MOSQLIMATE_API_KEY, never hardcoded).
    repository : "owner/repo_name" of the commit being submitted.
    commit : git commit hash of the model version used to produce `units`' forecasts.
    disease : "dengue" or "chikungunya".
    scope : "state" (adm_level=1) or "city" (adm_level=2).
    units : state codes (e.g. "AC") if scope="state", or geocodes (e.g. 1200401) if scope="city".
    case_definition : passed through to mosqlient (default "probable").
    published : whether the prediction should be public.
    results_dir : root of the results tree; defaults to ROOT_DIR/python_prophet/results.
    data_dir : root of processed_data for this disease; defaults to ROOT_DIR/processed_data/<disease>.
    target_ids : which target windows to submit (default: target_1..target_4).
    dry_run : if True, resolve everything but skip the actual upload call.

    Returns
    -------
    list of dicts describing each (unit, target) submission outcome.
    """
    if disease not in DISEASE_CODES:
        raise ValueError(f"Unknown disease '{disease}'. Expected one of {list(DISEASE_CODES)}.")
    if scope not in ("state", "city"):
        raise ValueError(f"Unknown scope '{scope}'. Expected 'state' or 'city'.")
    if not dry_run and not api_key:
        raise ValueError(
            "Missing API key. Set MOSQLIMATE_API_KEY in a .env file at the repo root."
        )

    disease_code = DISEASE_CODES[disease]
    adm_level = 1 if scope == "state" else 2

    data_dir = data_dir or (ROOT_DIR / "processed_data" / disease)
    results_dir = results_dir or (SRC_DIR.parent / "results")
    pred_dir = results_dir / disease / scope / "predictions"

    mosq = None
    if not dry_run:
        import mosqlient

        mosq = mosqlient

    results = []
    for unit in units:
        try:
            adm_1, adm_2 = _get_adm_codes(unit, scope, disease, data_dir)
        except FileNotFoundError as e:
            print(f"[SKIP] {unit}: processed-data file not found ({e.filename})")
            results.append({"unit": unit, "status": "skipped", "reason": "no adm codes"})
            continue

        for target_id in target_ids:
            pred_file = pred_dir / f"{unit}_{target_id}.csv"
            if not pred_file.exists():
                print(f"[SKIP] {unit}/{target_id}: prediction file not found ({pred_file})")
                results.append({"unit": unit, "target": target_id, "status": "skipped"})
                continue

            pred = pd.read_csv(pred_file)
            label = "state" if scope == "state" else "city"
            description = f"{disease.capitalize()} Prophet prediction for {label} {unit}, {target_id}"

            if dry_run:
                print(
                    f"[DRY-RUN] disease={disease_code} adm_level={adm_level} "
                    f"adm_1={adm_1} adm_2={adm_2} unit={unit} {target_id} "
                    f"rows={len(pred)} file={pred_file.name}"
                )
                results.append({"unit": unit, "target": target_id, "status": "dry-run"})
                continue

            outcome, reason = _upload_one(
                mosq,
                api_key=api_key,
                repository=repository,
                description=description,
                commit=commit,
                disease=disease_code,
                case_definition=case_definition,
                adm_level=adm_level,
                adm_1=adm_1,
                adm_2=adm_2,
                published=published,
                prediction=pred,
            )

            if outcome == "ok":
                print(f"[OK] {unit}/{target_id} submitted.")
                results.append({"unit": unit, "target": target_id, "status": "ok"})
            elif outcome == "skip_target":
                print(f"[SKIP] {unit}/{target_id}: {reason}; continuing with next target.")
                results.append({"unit": unit, "target": target_id, "status": "skipped", "reason": reason})
            else:  # skip_unit
                print(f"[SKIP] {unit}/{target_id}: {reason}; skipping remaining targets for {unit}.")
                results.append({"unit": unit, "target": target_id, "status": "skipped", "reason": reason})
                break

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit Prophet forecasts to Mosqlimate.")
    parser.add_argument("--disease", choices=["dengue", "chikungunya"], default="dengue")
    parser.add_argument("--scope", choices=["state", "city"], default="state")
    parser.add_argument(
        "--units",
        default="all",
        help="Comma-separated state codes or geocodes (default: all units with forecasts).",
    )
    parser.add_argument("--repository", required=True, help='"owner/repo_name" of the submitted commit.')
    parser.add_argument("--commit", required=True, help="Git commit hash of the model version.")
    parser.add_argument("--case-definition", default="probable")
    parser.add_argument("--published", action="store_true", default=True)
    parser.add_argument("--unpublished", dest="published", action="store_false")
    parser.add_argument("--dry-run", action="store_true", help="Resolve everything but skip the upload.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    units = (
        _default_units(args.disease, args.scope)
        if args.units.lower() == "all"
        else [u.strip() for u in args.units.split(",")]
    )
    if args.scope == "city":
        units = [int(u) for u in units]

    api_key = os.environ.get("MOSQLIMATE_API_KEY", "")

    results = submit_predictions(
        api_key=api_key,
        repository=args.repository,
        commit=args.commit,
        disease=args.disease,
        scope=args.scope,
        units=units,
        case_definition=args.case_definition,
        published=args.published,
        dry_run=args.dry_run,
    )

    ok = [r for r in results if r["status"] in ("ok", "dry-run")]
    skipped = [r for r in results if r["status"] == "skipped"]
    print(f"\nDone: {len(ok)} submitted, {len(skipped)} skipped.")


if __name__ == "__main__":
    main()
