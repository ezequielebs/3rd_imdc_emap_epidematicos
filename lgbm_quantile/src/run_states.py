import os
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


MODEL_NAME = "lgbm_quantile"
ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / MODEL_NAME / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
PRED_DIR = RESULTS_DIR / "pred"

ALL_STATES = [
    "AC", "AL", "AM", "AP", "BA", "CE", "DF", "GO", "MA",
    "MG", "MS", "MT", "PA", "PB", "PE", "PI", "PR", "RJ",
    "RN", "RO", "RR", "RS", "SC", "SE", "SP", "TO",
]

QUANTILES = [0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975]
QUANTILE_COLUMNS = {
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
LEVELS = [50, 80, 90, 95]
LAGS = [1, 2, 3, 4, 5, 6, 8, 13, 26, 39, 52, 104]
WINDOWS = [4, 8, 13, 26, 52]
CLIMATE_BASE_COLUMNS = [
    "temp_min_mean", "temp_med_mean", "temp_max_mean",
    "precip_min_mean", "precip_med_mean", "precip_max_mean",
    "pressure_min_mean", "pressure_med_mean", "pressure_max_mean",
    "rel_humid_min_mean", "rel_humid_med_mean", "rel_humid_max_mean",
    "thermal_range_mean", "rainy_days_mean", "enso", "iod", "pdo",
]
CLIMATE_LAG_COLUMNS = ["temp_med_mean", "precip_med_mean", "rel_humid_med_mean"]
ID_COLUMNS = ["disease", "uf"]
CATEGORICAL_COLUMNS = ["disease", "uf", "koppen", "biome"]
STATIC_NUMERIC_COLUMNS = ["uf_code", "pop", "log_pop"]
SEED = 42


def env_list(name, default):
    value = os.getenv(name, default)
    if value.lower() == "all":
        return ALL_STATES
    return [x.strip() for x in value.split(",") if x.strip()]


def disease_list():
    value = os.getenv("DISEASES", "dengue")
    return [x.strip() for x in value.split(",") if x.strip()]


def assert_not_lfs_pointer(path):
    with open(path, "rb") as f:
        prefix = f.read(64)
    if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise RuntimeError(
            f"{path} is still a Git LFS pointer. Run `git lfs pull` before the benchmark."
        )


def load_state_data(disease, states):
    pieces = []
    for state in states:
        path = ROOT / "processed_data" / disease / f"{disease}_{state}_agg.csv.gz"
        assert_not_lfs_pointer(path)
        df = pd.read_csv(path)
        df["disease"] = disease
        df["date"] = pd.to_datetime(df["date"])
        pieces.append(df)
    data = pd.concat(pieces, ignore_index=True)
    data["cases"] = data["cases"].clip(lower=0)
    data["epiweek_number"] = data["epiweek"] % 100
    data["log_pop"] = np.log1p(data["pop"])
    return data.sort_values(ID_COLUMNS + ["date"]).reset_index(drop=True)


def add_time_features(df):
    out = df.copy()
    ew = out["epiweek_number"].astype(float)
    out["season_year"] = np.where(out["epiweek_number"] >= 41, out["year"], out["year"] - 1)
    out["weeks_since_EW41"] = np.where(out["epiweek_number"] >= 41, out["epiweek_number"] - 41, out["epiweek_number"] + 12)
    out["weeks_until_EW40"] = np.where(out["epiweek_number"] <= 40, 40 - out["epiweek_number"], 92 - out["epiweek_number"])
    out["is_preseason_gap"] = out["epiweek_number"].between(26, 40).astype(int)
    out["is_high_season"] = out["epiweek_number"].between(1, 20).astype(int)
    out["sin_ew_52"] = np.sin(2 * np.pi * ew / 52)
    out["cos_ew_52"] = np.cos(2 * np.pi * ew / 52)
    out["sin_ew_26"] = np.sin(2 * np.pi * ew / 26)
    out["cos_ew_26"] = np.cos(2 * np.pi * ew / 26)
    return out


def add_history_features(df):
    out = add_time_features(df).sort_values(ID_COLUMNS + ["date"]).copy()
    group = out.groupby(ID_COLUMNS, sort=False)

    for lag in LAGS:
        out[f"lag_{lag}"] = group["cases"].shift(lag)

    shifted_cases = group["cases"].shift(1)
    for window in WINDOWS:
        out[f"rolling_mean_{window}"] = shifted_cases.groupby([out["disease"], out["uf"]]).rolling(window, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
        if window in [4, 13]:
            out[f"rolling_sum_{window}"] = shifted_cases.groupby([out["disease"], out["uf"]]).rolling(window, min_periods=1).sum().reset_index(level=[0, 1], drop=True)
        if window in [13, 26]:
            out[f"rolling_max_{window}"] = shifted_cases.groupby([out["disease"], out["uf"]]).rolling(window, min_periods=1).max().reset_index(level=[0, 1], drop=True)
            out[f"rolling_std_{window}"] = shifted_cases.groupby([out["disease"], out["uf"]]).rolling(window, min_periods=2).std().reset_index(level=[0, 1], drop=True)

    out["growth_1w"] = np.log1p(out["lag_1"]) - np.log1p(out["lag_2"])
    out["growth_4w"] = np.log1p(out["lag_1"]) - np.log1p(out["lag_5"])
    out["seasonal_ratio"] = out["lag_1"] / (out["lag_52"] + 1)
    out["recent_vs_seasonal"] = out["rolling_mean_4"] / (out["rolling_mean_52"] + 1)

    out["incidence"] = out["cases"] / out["pop"] * 100000
    shifted_incidence = group["incidence"].shift(1)
    out["incidence_lag_1"] = shifted_incidence
    out["incidence_lag_4"] = group["incidence"].shift(4)
    out["incidence_roll_4"] = shifted_incidence.groupby([out["disease"], out["uf"]]).rolling(4, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
    out["incidence_roll_13"] = shifted_incidence.groupby([out["disease"], out["uf"]]).rolling(13, min_periods=1).mean().reset_index(level=[0, 1], drop=True)

    for col in CLIMATE_LAG_COLUMNS:
        if col in out.columns:
            out[f"{col}_lag_1"] = group[col].shift(1)
            out[f"{col}_lag_4"] = group[col].shift(4)
            out[f"{col}_roll_4"] = group[col].shift(1).groupby([out["disease"], out["uf"]]).rolling(4, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
            out[f"{col}_roll_8"] = group[col].shift(1).groupby([out["disease"], out["uf"]]).rolling(8, min_periods=1).mean().reset_index(level=[0, 1], drop=True)

    return out


def build_current_feature_rows(work, current_date):
    current = work.loc[work["date"] == current_date].copy()
    current = add_time_features(current)
    history = work.loc[work["date"] < current_date].sort_values(ID_COLUMNS + ["date"])

    for idx, row in current.iterrows():
        mask = (history["disease"] == row["disease"]) & (history["uf"] == row["uf"])
        hist = history.loc[mask]
        cases = hist["cases"].to_numpy()

        for lag in LAGS:
            current.at[idx, f"lag_{lag}"] = cases[-lag] if len(cases) >= lag else np.nan

        for window in WINDOWS:
            values = cases[-window:]
            current.at[idx, f"rolling_mean_{window}"] = np.nanmean(values) if len(values) else np.nan
            if window in [4, 13]:
                current.at[idx, f"rolling_sum_{window}"] = np.nansum(values) if len(values) else np.nan
            if window in [13, 26]:
                current.at[idx, f"rolling_max_{window}"] = np.nanmax(values) if len(values) else np.nan
                current.at[idx, f"rolling_std_{window}"] = np.nanstd(values, ddof=1) if len(values) > 1 else np.nan

        current.at[idx, "growth_1w"] = np.log1p(current.at[idx, "lag_1"]) - np.log1p(current.at[idx, "lag_2"])
        current.at[idx, "growth_4w"] = np.log1p(current.at[idx, "lag_1"]) - np.log1p(current.at[idx, "lag_5"])
        current.at[idx, "seasonal_ratio"] = current.at[idx, "lag_1"] / (current.at[idx, "lag_52"] + 1)
        current.at[idx, "recent_vs_seasonal"] = current.at[idx, "rolling_mean_4"] / (current.at[idx, "rolling_mean_52"] + 1)

        pop = row["pop"]
        incidence = cases / pop * 100000 if len(cases) else np.array([])
        current.at[idx, "incidence_lag_1"] = incidence[-1] if len(incidence) >= 1 else np.nan
        current.at[idx, "incidence_lag_4"] = incidence[-4] if len(incidence) >= 4 else np.nan
        current.at[idx, "incidence_roll_4"] = np.nanmean(incidence[-4:]) if len(incidence) else np.nan
        current.at[idx, "incidence_roll_13"] = np.nanmean(incidence[-13:]) if len(incidence) else np.nan

        for col in CLIMATE_LAG_COLUMNS:
            if col in hist.columns:
                climate = hist[col].to_numpy()
                current.at[idx, f"{col}_lag_1"] = climate[-1] if len(climate) >= 1 else np.nan
                current.at[idx, f"{col}_lag_4"] = climate[-4] if len(climate) >= 4 else np.nan
                current.at[idx, f"{col}_roll_4"] = np.nanmean(climate[-4:]) if len(climate) else np.nan
                current.at[idx, f"{col}_roll_8"] = np.nanmean(climate[-8:]) if len(climate) else np.nan

    return current


def feature_columns(df):
    wanted = set(STATIC_NUMERIC_COLUMNS)
    wanted.update([
        "epiweek_number", "season_year", "weeks_since_EW41", "weeks_until_EW40",
        "is_preseason_gap", "is_high_season", "sin_ew_52", "cos_ew_52",
        "sin_ew_26", "cos_ew_26",
    ])
    wanted.update([c for c in CLIMATE_BASE_COLUMNS if c in df.columns])
    wanted.update([f"lag_{lag}" for lag in LAGS])
    wanted.update([f"rolling_mean_{window}" for window in WINDOWS])
    wanted.update(["rolling_sum_4", "rolling_sum_13"])
    wanted.update(["rolling_max_13", "rolling_max_26", "rolling_std_13", "rolling_std_26"])
    wanted.update(["growth_1w", "growth_4w", "seasonal_ratio", "recent_vs_seasonal"])
    wanted.update(["incidence_lag_1", "incidence_lag_4", "incidence_roll_4", "incidence_roll_13"])
    for col in CLIMATE_LAG_COLUMNS:
        wanted.update([f"{col}_lag_1", f"{col}_lag_4", f"{col}_roll_4", f"{col}_roll_8"])
    numeric_cols = [c for c in df.columns if c in wanted and pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in CATEGORICAL_COLUMNS if c in df.columns]
    return numeric_cols, categorical_cols


def make_matrix(df, numeric_cols, categorical_cols, columns=None):
    x_num = df[numeric_cols].copy()
    x_cat = pd.get_dummies(df[categorical_cols].astype("category"), dummy_na=True)
    x = pd.concat([x_num, x_cat], axis=1)
    x = x.replace([np.inf, -np.inf], np.nan)
    if columns is not None:
        x = x.reindex(columns=columns, fill_value=0)
    return x


def fit_quantile_models(x_train, y_train, n_estimators):
    models = {}
    for q in QUANTILES:
        model = LGBMRegressor(
            objective="quantile",
            alpha=q,
            n_estimators=n_estimators,
            learning_rate=0.03,
            num_leaves=64,
            min_child_samples=20,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=SEED,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(x_train, y_train)
        models[q] = model
    return models


def climatology_tables(train_df):
    train = train_df.copy()
    by_unit_ew = train.groupby(ID_COLUMNS + ["epiweek_number"])["cases"]
    by_disease_ew = train.groupby(["disease", "epiweek_number"])["cases"]
    by_unit = train.groupby(ID_COLUMNS)["cases"]
    return {
        "unit_ew_quantiles": by_unit_ew.quantile(QUANTILES).unstack().reset_index(),
        "disease_ew_quantiles": by_disease_ew.quantile(QUANTILES).unstack().reset_index(),
        "unit_median": by_unit.median().reset_index(name="unit_median"),
        "residual_quantiles": train.groupby("disease")["cases"].quantile(QUANTILES).unstack().reset_index(),
    }


def add_climatology_quantiles(rows, tables):
    out = rows[ID_COLUMNS + ["epiweek_number"]].copy()
    unit = tables["unit_ew_quantiles"]
    disease = tables["disease_ew_quantiles"]
    out = out.merge(unit, on=ID_COLUMNS + ["epiweek_number"], how="left")
    missing = out[QUANTILES].isna().any(axis=1)
    if missing.any():
        fallback = rows.loc[missing, ID_COLUMNS + ["epiweek_number"]].merge(
            disease, on=["disease", "epiweek_number"], how="left"
        )
        out.loc[missing, QUANTILES] = fallback[QUANTILES].to_numpy()
    out[QUANTILES] = out[QUANTILES].fillna(0)
    return out[QUANTILES].to_numpy()


def seasonal_naive_quantiles(rows, feature_rows, tables):
    base = feature_rows["lag_52"].fillna(feature_rows["rolling_mean_52"]).fillna(feature_rows["lag_26"]).fillna(0).to_numpy()
    residual = rows[["disease"]].merge(tables["residual_quantiles"], on="disease", how="left")[QUANTILES].fillna(0).to_numpy()
    median_residual = residual[:, QUANTILES.index(0.50)]
    spread = residual - median_residual[:, None]
    return np.maximum(base[:, None] + 0.25 * spread, 0)


def postprocess_quantiles(values):
    values = np.maximum(values, 0)
    return np.sort(values, axis=1)


def compute_metrics(pred_df):
    y = pred_df["actual"].to_numpy()
    median = pred_df["pred"].to_numpy()
    rows = {
        "wis": weighted_interval_score(pred_df),
        "mae": np.mean(np.abs(y - median)),
        "mse": np.mean((y - median) ** 2),
        "rmse": np.sqrt(np.mean((y - median) ** 2)),
        "mape": np.mean(np.abs((y - median) / np.maximum(y, 1))) * 100,
    }
    for level in LEVELS:
        rows[f"coverage_{level}"] = np.mean(
            (y >= pred_df[f"lower_{level}"].to_numpy()) &
            (y <= pred_df[f"upper_{level}"].to_numpy())
        )
    return rows


def weighted_interval_score(pred_df):
    y = pred_df["actual"].to_numpy()
    median = pred_df["pred"].to_numpy()
    score = 0.5 * np.abs(y - median)
    for level in LEVELS:
        alpha = 1 - level / 100
        lower = pred_df[f"lower_{level}"].to_numpy()
        upper = pred_df[f"upper_{level}"].to_numpy()
        interval_score = (upper - lower)
        interval_score += (2 / alpha) * (lower - y) * (y < lower)
        interval_score += (2 / alpha) * (y - upper) * (y > upper)
        score += (alpha / 2) * interval_score
    return np.mean(score / (len(LEVELS) + 0.5))


def quantile_frame(rows, values, model_name):
    out = rows[["disease", "uf", "date", "epiweek", "cases"]].copy()
    out = out.rename(columns={"cases": "actual"})
    out["model"] = model_name
    for i, q in enumerate(QUANTILES):
        out[QUANTILE_COLUMNS[q]] = values[:, i]
    return out


def replace_future_climate_with_climatology(work, train_rows, future_mask):
    cols = [c for c in CLIMATE_BASE_COLUMNS if c in work.columns]
    if not cols:
        return work
    unit_clim = train_rows.groupby(ID_COLUMNS + ["epiweek_number"])[cols].median().reset_index()
    disease_clim = train_rows.groupby(["disease", "epiweek_number"])[cols].median().reset_index()
    future = work.loc[future_mask, ID_COLUMNS + ["epiweek_number"]].copy()
    future = future.merge(unit_clim, on=ID_COLUMNS + ["epiweek_number"], how="left")
    missing = future[cols].isna().any(axis=1)
    if missing.any():
        fallback = work.loc[future_mask, ID_COLUMNS + ["epiweek_number"]].iloc[np.where(missing)[0]].merge(
            disease_clim, on=["disease", "epiweek_number"], how="left"
        )
        future.loc[missing, cols] = fallback[cols].to_numpy()
    work.loc[future_mask, cols] = future[cols].to_numpy()
    return work


def predict_fold(data, fold, n_estimators):
    train_col = f"train_{fold}"
    target_col = f"target_{fold}"
    train_mask = data[train_col].astype(bool)
    target_mask = data[target_col].astype(bool)
    cutoff_date = data.loc[train_mask, "date"].max()
    target_end = data.loc[target_mask, "date"].max()

    train_base = data.loc[train_mask].copy()
    train_features = add_history_features(train_base)
    numeric_cols, categorical_cols = feature_columns(train_features)
    x_train = make_matrix(train_features, numeric_cols, categorical_cols)
    y_train = np.log1p(train_features["cases"].to_numpy())
    models = fit_quantile_models(x_train, y_train, n_estimators)
    tables = climatology_tables(train_base)

    work = data.loc[data["date"] <= target_end].copy()
    future_mask = work["date"] > cutoff_date
    work.loc[future_mask, "cases"] = np.nan
    work = replace_future_climate_with_climatology(work, train_base, future_mask)

    prediction_pieces = []
    future_dates = sorted(work.loc[future_mask, "date"].unique())
    for current_date in future_dates:
        current_rows = work["date"] == current_date
        current_features = build_current_feature_rows(work, current_date)
        x_current = make_matrix(current_features, numeric_cols, categorical_cols, columns=x_train.columns)

        lgbm_values = np.column_stack([
            np.expm1(models[q].predict(x_current)) for q in QUANTILES
        ])
        lgbm_values = postprocess_quantiles(lgbm_values)

        rows = work.loc[current_rows].copy()
        clim_values = postprocess_quantiles(add_climatology_quantiles(rows, tables))
        seasonal_values = postprocess_quantiles(seasonal_naive_quantiles(rows, current_features, tables))
        ensemble_values = postprocess_quantiles(
            0.70 * lgbm_values + 0.20 * clim_values + 0.10 * seasonal_values
        )

        work.loc[current_rows, "cases"] = ensemble_values[:, QUANTILES.index(0.50)]

        target_rows = data["date"].eq(current_date) & data[target_col].astype(bool)
        if target_rows.any():
            actual_rows = data.loc[target_rows].copy()
            for model_name, values in [
                ("lgbm_quantile", lgbm_values),
                ("climatology", clim_values),
                ("seasonal_naive", seasonal_values),
                ("ensemble", ensemble_values),
            ]:
                prediction_pieces.append(quantile_frame(actual_rows, values, model_name))

    pred = pd.concat(prediction_pieces, ignore_index=True)
    pred["target_id"] = target_col
    return pred


def metrics_by_group(pred):
    rows = []
    group_cols = ["model", "disease", "uf", "target_id"]
    for keys, group in pred.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        row.update(compute_metrics(group))
        rows.append(row)
    metrics = pd.DataFrame(rows)

    aggregate_rows = []
    for keys, group in pred.groupby(["model", "disease", "target_id"]):
        row = {"model": keys[0], "disease": keys[1], "uf": "ALL", "target_id": keys[2]}
        row.update(compute_metrics(group))
        aggregate_rows.append(row)
    return pd.concat([metrics, pd.DataFrame(aggregate_rows)], ignore_index=True)


def mean_wis_by_target(df):
    return np.mean([weighted_interval_score(group) for _, group in df.groupby("target_id")])


def optimized_ensemble(pred):
    qcols = [QUANTILE_COLUMNS[q] for q in QUANTILES]
    key_cols = ["disease", "uf", "date", "epiweek", "actual", "target_id"]
    base_models = ["lgbm_quantile", "climatology", "seasonal_naive"]
    parts = {
        model: pred[pred["model"] == model].sort_values(key_cols).reset_index(drop=True)
        for model in base_models
    }
    base = parts["lgbm_quantile"][key_cols].copy()

    best_score = np.inf
    best_weights = None
    best_values = None
    for i_lgbm in range(21):
        w_lgbm = i_lgbm / 20
        for i_clim in range(21 - i_lgbm):
            w_clim = i_clim / 20
            w_seasonal = 1 - w_lgbm - w_clim
            values = (
                w_lgbm * parts["lgbm_quantile"][qcols].to_numpy()
                + w_clim * parts["climatology"][qcols].to_numpy()
                + w_seasonal * parts["seasonal_naive"][qcols].to_numpy()
            )
            values = postprocess_quantiles(values)
            candidate = base.copy()
            for j, col in enumerate(qcols):
                candidate[col] = values[:, j]
            score = mean_wis_by_target(candidate)
            if score < best_score:
                best_score = score
                best_weights = (w_lgbm, w_clim, w_seasonal)
                best_values = values

    out = base.copy()
    out["model"] = "optimized_ensemble"
    for j, col in enumerate(qcols):
        out[col] = best_values[:, j]
    out = out[["disease", "uf", "date", "epiweek", "actual", "model"] + qcols + ["target_id"]]
    weights = pd.DataFrame([{
        "model": "optimized_ensemble",
        "w_lgbm_quantile": best_weights[0],
        "w_climatology": best_weights[1],
        "w_seasonal_naive": best_weights[2],
        "validation_mean_wis": best_score,
    }])
    return out, weights


def run_disease(disease, states, n_estimators):
    data = load_state_data(disease, states)
    fold_predictions = []
    for fold in range(1, 5):
        if data[f"target_{fold}"].astype(bool).sum() == 0:
            continue
        print(f"{disease}: fold {fold}")
        fold_predictions.append(predict_fold(data, fold, n_estimators))

    pred = pd.concat(fold_predictions, ignore_index=True)
    opt_pred, opt_weights = optimized_ensemble(pred)
    pred = pd.concat([pred, opt_pred], ignore_index=True)
    metrics = metrics_by_group(pred)
    selection = (
        metrics[metrics["uf"] == "ALL"]
        .groupby(["model", "disease"], as_index=False)
        .agg(mean_wis=("wis", "mean"), mean_mae=("mae", "mean"), mean_rmse=("rmse", "mean"))
        .sort_values("mean_wis")
    )
    best_model = selection.iloc[0]["model"]
    benchmark_pred = pred[pred["model"] == best_model].copy()
    benchmark_pred["selected_by"] = "lowest_validation_wis"

    pred.to_csv(PRED_DIR / f"predictions_{disease}.csv.gz", index=False)
    benchmark_pred.to_csv(PRED_DIR / f"benchmark_predictions_{disease}.csv.gz", index=False)
    metrics.to_csv(METRICS_DIR / f"metrics_{disease}.csv", index=False)
    selection.to_csv(METRICS_DIR / f"model_selection_{disease}.csv", index=False)
    opt_weights.to_csv(METRICS_DIR / f"optimized_ensemble_weights_{disease}.csv", index=False)
    return metrics


def main():
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    diseases = disease_list()
    states = env_list("STATES", "all")
    n_estimators = int(os.getenv("N_ESTIMATORS", "1000"))

    all_metrics = []
    for disease in diseases:
        all_metrics.append(run_disease(disease, states, n_estimators))

    comparison = pd.concat(all_metrics, ignore_index=True)
    comparison.to_csv(METRICS_DIR / "model_comparison.csv", index=False)

    summary = (
        comparison[comparison["uf"] == "ALL"]
        .groupby(["model", "disease"], as_index=False)
        .agg(
            mean_wis=("wis", "mean"),
            mean_mae=("mae", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_coverage_50=("coverage_50", "mean"),
            mean_coverage_80=("coverage_80", "mean"),
            mean_coverage_90=("coverage_90", "mean"),
            mean_coverage_95=("coverage_95", "mean"),
        )
        .sort_values(["disease", "mean_wis"])
    )
    summary.to_csv(METRICS_DIR / "benchmark_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
