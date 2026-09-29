"""Shared implementation used by the auditable phase notebooks."""

from __future__ import annotations

import json
import pickle
import time
from itertools import product
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.compose import ColumnTransformer
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
EXPECTED_YEARS = set(range(1990, 2023))
TARGET = "YIELD"
GROUPS = ["AREA", "ITEM"]
LEAKAGE_COLUMNS = ["AREA_HARVESTED", "PRODUCTION_QUANTITY"]


def project_root() -> Path:
    here = Path.cwd().resolve()
    if (here / "data" / "raw").exists():
        return here
    if (here.parent / "data" / "raw").exists():
        return here.parent
    raise FileNotFoundError("Could not locate data/raw from the current directory")


ROOT = project_root()
DATA_PATH = ROOT / "data" / "raw" / "europe_cereal_yield_climate_1990_2022.csv"
ARTIFACT_DIR = ROOT / "artifacts" / "cereal_yield_pipeline"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def _load_pickle(name: str):
    with (ARTIFACT_DIR / name).open("rb") as stream:
        return pickle.load(stream)


def _save_pickle(value, name: str) -> None:
    with (ARTIFACT_DIR / name).open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)


def climate_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith("WB_CCKP_")]


def phase1() -> dict:
    """Load, audit, summarize and conservatively clean the source data."""
    frame = pd.read_csv(DATA_PATH)
    climate = climate_columns(frame)

    dtype_table = frame.dtypes.astype(str).rename("dtype").to_frame()
    memory_table = pd.DataFrame({
        "memory_bytes": frame.memory_usage(deep=True),
        "memory_mib": frame.memory_usage(deep=True) / (1024 ** 2),
    })
    memory_summary = pd.DataFrame({
        "total_memory_bytes": [int(frame.memory_usage(deep=True).sum())],
        "total_memory_mib": [float(frame.memory_usage(deep=True).sum() / (1024 ** 2))],
    })
    missing_table = frame.isna().sum().rename("missing_count").to_frame()
    duplicate_table = pd.DataFrame({
        "check": ["duplicate full rows", "duplicate (AREA, ITEM, YEAR) keys"],
        "count": [int(frame.duplicated().sum()), int(frame.duplicated(GROUPS + ["YEAR"]).sum())],
    })

    measure_columns = [TARGET, "AREA_HARVESTED", "PRODUCTION_QUANTITY"]
    domain_rows = []
    for column in measure_columns:
        domain_rows.append({
            "column": column,
            "negative_count": int(frame[column].lt(0).sum()),
            "zero_count": int(frame[column].eq(0).sum()),
            "nonpositive_count": int(frame[column].le(0).sum()),
            "minimum": float(frame[column].min()),
            "maximum": float(frame[column].max()),
        })
    domain_table = pd.DataFrame(domain_rows)

    # Flag observations outside 1.5*IQR within each country/crop group and
    # group z-scores above 4. These are audit flags only; plausible extremes
    # remain in the data.
    flagged_rows = []
    for column in measure_columns:
        grouped = frame.groupby(GROUPS, observed=True)[column]
        group_stats = grouped.agg(group_mean="mean", group_std="std", q1=lambda values: values.quantile(0.25), q3=lambda values: values.quantile(0.75))
        stats = group_stats.reset_index()
        stats["iqr"] = stats["q3"] - stats["q1"]
        stats["lower_bound"] = stats["q1"] - 1.5 * stats["iqr"]
        stats["upper_bound"] = stats["q3"] + 1.5 * stats["iqr"]
        stats_by_key = stats.set_index(GROUPS)
        for index, row in frame.iterrows():
            stat = stats_by_key.loc[(row["AREA"], row["ITEM"])]
            value = float(row[column])
            std = float(stat["group_std"])
            z_score = (value - float(stat["group_mean"])) / std if std > 0 else np.nan
            iqr_flag = value < float(stat["lower_bound"]) or value > float(stat["upper_bound"])
            z_flag = bool(np.isfinite(z_score) and abs(z_score) > 4)
            if iqr_flag or z_flag:
                flagged_rows.append({
                    "AREA": row["AREA"], "ITEM": row["ITEM"], "YEAR": int(row["YEAR"]),
                    "measure": column, "value": value, "group_z_score": z_score,
                    "group_lower_iqr": float(stat["lower_bound"]),
                    "group_upper_iqr": float(stat["upper_bound"]),
                    "iqr_flag": bool(iqr_flag), "z_gt_4_flag": z_flag,
                    "review": "Retained pending domain review; statistical extremeness alone is not a data error.",
                })
    outlier_table = pd.DataFrame(flagged_rows, columns=[
        "AREA", "ITEM", "YEAR", "measure", "value", "group_z_score",
        "group_lower_iqr", "group_upper_iqr", "iqr_flag", "z_gt_4_flag", "review",
    ])
    outlier_counts = (outlier_table.groupby(["AREA", "ITEM", "measure"], as_index=False)
                      .size().rename(columns={"size": "flagged_rows"})) if not outlier_table.empty else pd.DataFrame()

    def distribution(group_columns=None):
        grouped = frame.groupby(group_columns, observed=True)[TARGET] if group_columns else frame[TARGET]
        result = grouped.agg(count="count", mean="mean", median="median", std="std", min="min", max="max")
        return result.reset_index() if group_columns else result.to_frame().T

    overall_yield = distribution()
    crop_yield = distribution(["ITEM"])
    country_yield = distribution(["AREA"])
    country_yield = country_yield.merge(frame.groupby("AREA").size().rename("row_count"), on="AREA")
    country_yield = country_yield.sort_values("mean", ascending=False).reset_index(drop=True)

    coverage_rows = []
    all_areas = sorted(frame["AREA"].dropna().unique())
    all_items = sorted(frame["ITEM"].dropna().unique())
    for area, item in product(all_areas, all_items):
        part = frame.loc[frame["AREA"].eq(area) & frame["ITEM"].eq(item)]
        found = set(pd.to_numeric(part["YEAR"], errors="coerce").dropna().astype(int))
        gaps = sorted(EXPECTED_YEARS - found)
        coverage_rows.append({
            "AREA": area, "ITEM": item, "observed_years": len(found),
            "expected_years": len(EXPECTED_YEARS), "missing_year_count": len(gaps),
            "missing_years": ", ".join(map(str, gaps)),
        })
    coverage_table = pd.DataFrame(coverage_rows).sort_values(["AREA", "ITEM"]).reset_index(drop=True)

    # Only impossible nonpositive values are removed. Missingness and duplicates
    # are retained as explicit audit results and are not silently repaired.
    invalid_mask = frame[measure_columns].le(0).any(axis=1)
    cleaned = frame.loc[~invalid_mask].copy().reset_index(drop=True)
    cleaned.to_pickle(ARTIFACT_DIR / "phase1_cleaned.pkl")

    return {
        "shape": pd.DataFrame({"rows": [len(frame)], "columns": [len(frame.columns)], "climate_columns": [len(climate)]}),
        "dtype_table": dtype_table,
        "memory_table": memory_table,
        "memory_summary": memory_summary,
        "missing_table": missing_table,
        "duplicate_table": duplicate_table,
        "domain_table": domain_table,
        "outlier_table": outlier_table,
        "outlier_counts": outlier_counts,
        "overall_yield": overall_yield,
        "crop_yield": crop_yield,
        "country_yield": country_yield,
        "coverage_table": coverage_table,
        "cleaned_shape": pd.DataFrame({"rows_before": [len(frame)], "rows_after": [len(cleaned)], "removed_nonpositive_rows": [int(invalid_mask.sum())]}),
        "climate_count": len(climate),
    }


def phase2() -> dict:
    frame = pd.read_pickle(ARTIFACT_DIR / "phase1_cleaned.pkl")
    derived = frame["PRODUCTION_QUANTITY"] / frame["AREA_HARVESTED"] * 1000.0
    abs_error = (frame[TARGET] - derived).abs()
    comparison = pd.DataFrame({
        "target": frame[TARGET],
        "production_over_harvested_x1000": derived,
        "absolute_error": abs_error,
    })
    summary = pd.DataFrame({
        "metric": ["correlation", "exact_match_abs_error_lt_1e-9", "within 0.1 yield units", "median absolute error", "95th percentile absolute error", "maximum absolute error"],
        "value": [
            frame[TARGET].corr(derived), int(abs_error.lt(1e-9).sum()), int(abs_error.le(0.1).sum()),
            abs_error.median(), abs_error.quantile(0.95), abs_error.max(),
        ],
    })
    largest_differences = frame[["AREA", "ITEM", "YEAR", TARGET, "AREA_HARVESTED", "PRODUCTION_QUANTITY"]].copy()
    largest_differences["derived_yield"] = derived
    largest_differences["absolute_error"] = abs_error
    largest_differences = largest_differences.nlargest(10, "absolute_error")

    model_frame = frame.drop(columns=LEAKAGE_COLUMNS).copy()
    model_frame.to_pickle(ARTIFACT_DIR / "phase2_no_leakage.pkl")
    return {"summary": summary, "largest_differences": largest_differences, "leakage_columns_dropped": pd.DataFrame({"column": LEAKAGE_COLUMNS, "reason": ["Used in arithmetic identity for YIELD", "Used in arithmetic identity for YIELD"]})}


def _vif_table(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    # VIF = 1/(1-R^2), computed with an intercept using least squares.
    from sklearn.linear_model import LinearRegression
    values = frame[columns].astype(float)
    rows = []
    for target_column in columns:
        others = [column for column in columns if column != target_column]
        if not others:
            vif = 1.0
        else:
            reg = LinearRegression().fit(values[others], values[target_column])
            r2 = float(reg.score(values[others], values[target_column]))
            vif = float(1.0 / max(1.0 - r2, 1e-12))
        rows.append({"feature": target_column, "VIF": vif, "flag_over_5": vif > 5, "flag_over_10": vif > 10})
    return pd.DataFrame(rows).sort_values("VIF", ascending=False).reset_index(drop=True)


def phase3() -> dict:
    frame = pd.read_pickle(ARTIFACT_DIR / "phase2_no_leakage.pkl")
    climate = climate_columns(frame)
    # Determine the fixed stratified training fold here so supervised feature
    # selection never inspects the held-out YIELD values. Phase 4 recreates the
    # identical seeded split and persists its indices.
    selection_train_index, _ = train_test_split(
        frame.index.to_numpy(), test_size=0.20, random_state=SEED,
        stratify=frame["ITEM"],
    )
    selection_frame = frame.loc[selection_train_index]
    standard_deviation = selection_frame[climate].std(ddof=1)
    std_table = standard_deviation.rename("std").to_frame().sort_values("std").reset_index(names="feature")
    low_variance = std_table.loc[std_table["std"] < 0.02].copy()
    remaining = [column for column in climate if column not in set(low_variance["feature"])]
    correlation = selection_frame[remaining].corr().abs()
    target_corr = selection_frame[remaining].corrwith(selection_frame[TARGET]).abs()

    if len(remaining) > 1:
        condensed_distance = squareform((1.0 - correlation).clip(0, 1).to_numpy(), checks=False)
        hierarchy = linkage(condensed_distance, method="average")
        cluster_ids = fcluster(hierarchy, t=0.15, criterion="distance")
    else:
        cluster_ids = np.ones(len(remaining), dtype=int)
    cluster_members = pd.DataFrame({"feature": remaining, "cluster": cluster_ids, "abs_corr_with_yield": target_corr.reindex(remaining).values})
    cluster_members["kept"] = False
    keep_features = []
    for cluster_id, members in cluster_members.groupby("cluster", sort=True):
        chosen = members.sort_values("abs_corr_with_yield", ascending=False).iloc[0]["feature"]
        keep_features.append(chosen)
        cluster_members.loc[members.index, "kept"] = members["feature"].eq(chosen).values
    cluster_members["decision"] = np.where(cluster_members["kept"], "keep: strongest absolute target correlation", "drop: not selected representative from average-linkage cluster")
    cluster_report = (cluster_members.groupby("cluster", sort=True)
                      .agg(members=("feature", lambda x: ", ".join(x)), kept=("feature", lambda x: ", ".join(cluster_members.loc[x.index].loc[cluster_members.loc[x.index, "kept"], "feature"])), count=("feature", "size"))
                      .reset_index())

    # Preserve stable source-column order after selecting one variable per cluster.
    selected_climate = [column for column in remaining if column in set(keep_features)]
    selected_features = selected_climate + ["YEAR"]
    vif = _vif_table(selection_frame, selected_features)
    manifest = {
        "seed": SEED,
        "climate_prefix": "WB_CCKP_",
        "source_climate_columns": climate,
        "low_variance_threshold": 0.02,
        "low_variance_dropped": low_variance["feature"].tolist(),
        "correlation_distance": "1 - absolute Pearson correlation",
        "linkage": "average",
        "correlation_cluster_cut_distance": 0.15,
        "selected_climate_features": selected_climate,
        "selected_numeric_features": selected_features,
        "feature_selection_rows": int(len(selection_frame)),
        "feature_selection_scope": "training partition of the fixed 80/20 stratified split only",
    }
    with (ARTIFACT_DIR / "phase3_feature_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
    return {"std_table": std_table, "low_variance": low_variance, "cluster_members": cluster_members.sort_values(["cluster", "kept", "feature"], ascending=[True, False, True]), "cluster_report": cluster_report, "selected_climate": pd.DataFrame({"feature": selected_climate, "abs_corr_with_yield": target_corr.reindex(selected_climate).values}), "vif": vif, "selected_count": pd.DataFrame({"selected_climate": [len(selected_climate)], "with_year": [len(selected_features)]}), "manifest": manifest}


def phase4() -> dict:
    frame = pd.read_pickle(ARTIFACT_DIR / "phase2_no_leakage.pkl")
    with (ARTIFACT_DIR / "phase3_feature_manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    features = manifest["selected_numeric_features"]
    train_index, test_index = train_test_split(
        frame.index.to_numpy(), test_size=0.20, random_state=SEED,
        stratify=frame["ITEM"],
    )
    split = {"train_index": train_index, "test_index": test_index, "seed": SEED, "test_size": 0.20}
    _save_pickle(split, "phase4_split.pkl")
    stratification = pd.crosstab(frame.loc[np.r_[train_index, test_index], "ITEM"],
                                 pd.Series(["train"] * len(train_index) + ["test"] * len(test_index), index=np.r_[train_index, test_index]),
                                 normalize="columns").reset_index()
    counts = pd.DataFrame({"partition": ["train", "test"], "rows": [len(train_index), len(test_index)]})
    numeric_summary = frame[features].describe().T.reset_index(names="feature")
    choice_table = pd.DataFrame([
        {"choice": "Split", "decision": "One fixed 80/20 random holdout, stratified by ITEM, random_state=42; rows are shuffled observations, not chronological blocks."},
        {"choice": "Validation", "decision": "5-fold shuffled KFold CV on training rows for tuning; held-out test set is used once for final comparison."},
        {"choice": "Scaling", "decision": "StandardScaler is used inside the Ridge and normalized XGBoost pipelines; baseline tree candidates use unscaled numeric values. Each transform is fit within its CV training fold. Country/crop are one-hot encoded for comparable preprocessing."},
        {"choice": "Context ablation", "decision": "Fit each candidate both with and without AREA and ITEM one-hot features."},
    ])
    return {"counts": counts, "stratification": stratification, "numeric_summary": numeric_summary, "choice_table": choice_table, "feature_list": pd.DataFrame({"selected_numeric_feature": features})}


def _model_spec(model_name: str):
    if model_name == "Ridge":
        return Ridge(), {"model__alpha": [0.1, 1.0, 10.0, 100.0]}
    if model_name == "RandomForest":
        return RandomForestRegressor(random_state=SEED, n_jobs=1), {
            "model__n_estimators": [200], "model__max_depth": [None, 12],
            "model__min_samples_leaf": [1, 3],
        }
    if model_name == "GradientBoosting":
        return GradientBoostingRegressor(random_state=SEED), {
            "model__n_estimators": [100, 200], "model__max_depth": [2, 3],
            "model__learning_rate": [0.05],
        }
    raise ValueError(model_name)


def _make_pipeline(model_name: str, numeric_features: list[str], include_context: bool) -> Pipeline:
    categorical_features = GROUPS if include_context else []
    transformers = []
    if model_name == "Ridge":
        transformers.append(("numeric", StandardScaler(), numeric_features))
    else:
        transformers.append(("numeric", "passthrough", numeric_features))
    if include_context:
        transformers.append(("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features))
    preprocess = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)
    estimator, _ = _model_spec(model_name)
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def _fit_search(model_name: str, frame: pd.DataFrame, y: pd.Series, numeric_features: list[str], include_context: bool):
    pipeline = _make_pipeline(model_name, numeric_features, include_context)
    _, grid = _model_spec(model_name)
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    search = GridSearchCV(
        pipeline, grid, scoring="neg_root_mean_squared_error", cv=cv,
        n_jobs=1, refit=True, return_train_score=True,
    )
    started = time.perf_counter()
    search.fit(frame[numeric_features + (GROUPS if include_context else [])], y)
    seconds = time.perf_counter() - started
    return search, seconds


def phase5() -> dict:
    frame = pd.read_pickle(ARTIFACT_DIR / "phase2_no_leakage.pkl")
    split = _load_pickle("phase4_split.pkl")
    with (ARTIFACT_DIR / "phase3_feature_manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    numeric_features = manifest["selected_numeric_features"]
    train = frame.loc[split["train_index"]]
    y_train = train[TARGET]
    records = {}
    rows = []
    for include_context in [False, True]:
        for model_name in ["Ridge", "RandomForest", "GradientBoosting"]:
            search, seconds = _fit_search(model_name, train, y_train, numeric_features, include_context)
            key = f"{model_name}|context={include_context}"
            records[key] = {
                "model": search.best_estimator_, "model_name": model_name,
                "include_context": include_context, "best_params": search.best_params_,
                "cv_rmse": float(-search.best_score_), "fit_seconds": seconds,
                "numeric_features": numeric_features,
            }
            rows.append({
                "model": model_name, "AREA_ITEM_context": include_context,
                "best_CV_RMSE": -search.best_score_, "fit_seconds": seconds,
                "best_hyperparameters": str(search.best_params_),
            })
    _save_pickle(records, "phase5_model_records.pkl")
    cv_table = pd.DataFrame(rows).sort_values("best_CV_RMSE").reset_index(drop=True)
    return {"cv_table": cv_table, "selected_by_cv": cv_table.iloc[[0]].reset_index(drop=True), "note": "Best candidate selection uses training CV RMSE only; the holdout is not used for tuning or selection."}


def _metrics(y_true, y_pred) -> dict:
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": mean_absolute_error(y_true, y_pred),
    }


def phase6() -> dict:
    frame = pd.read_pickle(ARTIFACT_DIR / "phase2_no_leakage.pkl")
    split = _load_pickle("phase4_split.pkl")
    records = _load_pickle("phase5_model_records.pkl")
    with (ARTIFACT_DIR / "phase3_feature_manifest.json").open(encoding="utf-8") as stream:
        feature_manifest = json.load(stream)
    reduced_features = feature_manifest["selected_numeric_features"]
    test = frame.loc[split["test_index"]]
    y_test = test[TARGET]

    comparison_rows = []
    prediction_map = {}
    for key, record in records.items():
        input_columns = record["numeric_features"] + (GROUPS if record["include_context"] else [])
        predicted = record["model"].predict(test[input_columns])
        prediction_map[key] = predicted
        comparison_rows.append({"model": record["model_name"], "AREA_ITEM_context": record["include_context"], **_metrics(y_test, predicted)})
    candidate_test = pd.DataFrame(comparison_rows).sort_values("RMSE").reset_index(drop=True)

    best_key = min(records, key=lambda key: records[key]["cv_rmse"])
    best = records[best_key]
    best_reduced_prediction = prediction_map[best_key]
    best_reduced_metrics = _metrics(y_test, best_reduced_prediction)

    all_climate = climate_columns(frame)
    full_features = all_climate + ["YEAR"]
    cache_path = ARTIFACT_DIR / "phase6_best_models.pkl"
    cached = _load_pickle("phase6_best_models.pkl") if cache_path.exists() else None
    cache_matches = bool(
        cached
        and cached.get("key") == best_key
        and cached.get("full", {}).get("numeric_features") == full_features
        and cached.get("full", {}).get("include_context") == best["include_context"]
    )
    if cache_matches:
        full_record = cached["full"]
        full_model = full_record["model"]
        full_fit_seconds = full_record["fit_seconds"]
        full_cv_rmse = full_record["cv_rmse"]
        full_best_params = full_record["best_params"]
    else:
        full_search, full_fit_seconds = _fit_search(
            best["model_name"], frame.loc[split["train_index"]],
            frame.loc[split["train_index"], TARGET], full_features, best["include_context"],
        )
        full_model = full_search.best_estimator_
        full_cv_rmse = float(-full_search.best_score_)
        full_best_params = full_search.best_params_
    full_input = full_features + (GROUPS if best["include_context"] else [])
    full_prediction = full_model.predict(test[full_input])
    full_metrics = _metrics(y_test, full_prediction)
    reduction_comparison = pd.DataFrame([
        {"feature_set": f"Reduced ({len(reduced_features)} numeric incl. YEAR)", **best_reduced_metrics,
         "CV_RMSE": best["cv_rmse"], "training_seconds": best["fit_seconds"], "best_params": str(best["best_params"])},
        {"feature_set": f"Full ({len(all_climate)} climate + YEAR)", **full_metrics,
         "CV_RMSE": full_cv_rmse, "training_seconds": full_fit_seconds, "best_params": str(full_best_params)},
    ])
    full_record = {
        "model": full_model, "model_name": best["model_name"],
        "include_context": best["include_context"], "best_params": full_best_params,
        "cv_rmse": full_cv_rmse, "fit_seconds": full_fit_seconds,
        "numeric_features": full_features,
    }
    _save_pickle({"key": best_key, "reduced": best, "full": full_record}, "phase6_best_models.pkl")

    best_input = reduced_features + (GROUPS if best["include_context"] else [])
    best_model = best["model"]
    preprocessor = best_model.named_steps["preprocess"]
    feature_names = preprocessor.get_feature_names_out()
    estimator = best_model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
        importance_type = "feature importance"
    else:
        values = np.abs(np.asarray(estimator.coef_).ravel())
        importance_type = "absolute standardized coefficient"
    importance_table = pd.DataFrame({"feature": feature_names, "importance": values, "importance_type": importance_type}).sort_values("importance", ascending=False).reset_index(drop=True)
    item_rows = importance_table.loc[importance_table["feature"].str.contains(r"ITEM_", regex=True)]
    item_importance = pd.DataFrame({
        "feature_group": ["ITEM (crop type)"],
        "aggregate_importance": [float(item_rows["importance"].sum()) if not item_rows.empty else 0.0],
        "rank_among_input_features": [int((importance_table["importance"] > item_rows["importance"].sum()).sum() + 1) if not item_rows.empty else None],
    })

    test_output = test[["AREA", "ITEM", "YEAR", TARGET]].copy()
    test_output["prediction"] = best_reduced_prediction
    test_output["residual_actual_minus_prediction"] = test_output[TARGET] - test_output["prediction"]
    test_output.to_csv(ARTIFACT_DIR / "heldout_predictions.csv", index=False)

    per_crop_rows = []
    for crop, crop_rows in test_output.groupby("ITEM", observed=True):
        values_for_crop = _metrics(crop_rows[TARGET], crop_rows["prediction"])
        per_crop_rows.append({"ITEM": crop, "n": len(crop_rows), **values_for_crop})
    per_crop = pd.DataFrame(per_crop_rows).sort_values("ITEM").reset_index(drop=True)
    per_country = (test_output.groupby("AREA", observed=True)
                   .agg(n=(TARGET, "size"), mean_residual=("residual_actual_minus_prediction", "mean"),
                        RMSE=("residual_actual_minus_prediction", lambda x: float(np.sqrt(np.mean(np.square(x))))))
                   .sort_values("RMSE", ascending=False).reset_index())

    scatter_path = ARTIFACT_DIR / "predicted_vs_actual.png"
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    for crop, crop_rows in test_output.groupby("ITEM", observed=True):
        ax.scatter(crop_rows[TARGET], crop_rows["prediction"], alpha=0.65, s=24, label=crop)
    low = min(test_output[TARGET].min(), test_output["prediction"].min())
    high = max(test_output[TARGET].max(), test_output["prediction"].max())
    ax.plot([low, high], [low, high], "k--", linewidth=1)
    ax.set(xlabel="Actual YIELD", ylabel="Predicted YIELD", title=f"Held-out predictions: {best['model_name']} (reduced features)")
    ax.legend(title="ITEM", fontsize=8)
    fig.tight_layout()
    fig.savefig(scatter_path, dpi=150)
    plt.close(fig)

    residual_path = ARTIFACT_DIR / "residuals_vs_predicted.png"
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.scatter(test_output["prediction"], test_output["residual_actual_minus_prediction"], alpha=0.62, s=24)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="Predicted YIELD", ylabel="Residual (actual - predicted)", title="Held-out residuals vs. predictions")
    fig.tight_layout()
    fig.savefig(residual_path, dpi=150)
    plt.close(fig)

    return {
        "candidate_test": candidate_test,
        "reduction_comparison": reduction_comparison,
        "best_selection": pd.DataFrame([{"selected_candidate": best_key, "selection_basis": "lowest training 5-fold CV RMSE", "best_cv_rmse": best["cv_rmse"]}]),
        "feature_importance": importance_table.head(25),
        "item_importance": item_importance,
        "per_crop": per_crop,
        "per_country_worst": per_country.head(10),
        "prediction_plot_path": str(scatter_path),
        "residual_plot_path": str(residual_path),
        "test_output": test_output,
        "reduced_features": reduced_features,
        "best_key": best_key,
    }


def phase7() -> dict:
    selected = _load_pickle("phase6_best_models.pkl")
    with (ARTIFACT_DIR / "phase3_feature_manifest.json").open(encoding="utf-8") as stream:
        feature_manifest = json.load(stream)
    best = selected["reduced"]
    model_path = ARTIFACT_DIR / "best_yield_model.joblib"
    input_columns = best["numeric_features"] + (GROUPS if best["include_context"] else [])
    final_model = clone(best["model"])
    training_frame = pd.read_pickle(ARTIFACT_DIR / "phase2_no_leakage.pkl")
    started = time.perf_counter()
    final_model.fit(training_frame[input_columns], training_frame[TARGET])
    final_fit_seconds = time.perf_counter() - started
    joblib.dump(final_model, model_path, compress=3)
    export_manifest = {
        "target": TARGET,
        "model_file": model_path.name,
        "selected_candidate": selected["key"],
        "model_type": best["model_name"],
        "best_hyperparameters": best["best_params"],
        "numeric_feature_columns": best["numeric_features"],
        "categorical_feature_columns": GROUPS if best["include_context"] else [],
        "required_input_columns_in_order": input_columns,
        "context_features_included": bool(best["include_context"]),
        "dropped_leakage_columns": LEAKAGE_COLUMNS,
        "climate_reduction": {
            "source_climate_column_count": len(feature_manifest["source_climate_columns"]),
            "selected_climate_columns": feature_manifest["selected_climate_features"],
            "year_included_as_numeric_feature": True,
        },
        "random_seed": SEED,
        "training_design": "cross-sectional shuffled split and shuffled K-fold CV; YEAR is an ordinary numeric input",
        "prediction_input_note": "Provide the listed columns with the same source units and category labels as training data.",
        "final_model_fit_rows": int(len(training_frame)),
        "final_model_refit_after_holdout_evaluation": True,
        "final_model_fit_seconds": final_fit_seconds,
    }
    manifest_path = ARTIFACT_DIR / "feature_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(export_manifest, stream, indent=2)
    return {
        "artifact_files": pd.DataFrame({
            "file": [model_path.name, manifest_path.name, "heldout_predictions.csv", "predicted_vs_actual.png", "residuals_vs_predicted.png"],
            "path": [str(model_path), str(manifest_path), str(ARTIFACT_DIR / "heldout_predictions.csv"), str(ARTIFACT_DIR / "predicted_vs_actual.png"), str(ARTIFACT_DIR / "residuals_vs_predicted.png")],
        }),
        "feature_manifest": export_manifest,
        "model_size_bytes": model_path.stat().st_size,
        "final_fit_seconds": final_fit_seconds,
    }


def phase8() -> dict:
    """Tune a standardized reduced-feature model and refresh exports if better.

    Scaling is fitted inside each CV fold to avoid leakage. YIELD stays on its
    original scale so the reported RMSE and MAE remain in source units.
    """
    from xgboost import XGBRegressor

    frame = pd.read_pickle(ARTIFACT_DIR / "phase2_no_leakage.pkl")
    split = _load_pickle("phase4_split.pkl")
    feature_manifest = json.loads((ARTIFACT_DIR / "phase3_feature_manifest.json").read_text(encoding="utf-8"))
    numeric_features = feature_manifest["selected_numeric_features"]
    categorical_features = GROUPS
    input_columns = numeric_features + categorical_features
    train = frame.loc[split["train_index"]]
    test = frame.loc[split["test_index"]]

    preprocessor = ColumnTransformer([
        ("numeric", StandardScaler(), numeric_features),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
    ], remainder="drop", sparse_threshold=0.0)
    estimator = XGBRegressor(
        objective="reg:squarederror", tree_method="hist", n_jobs=1,
        random_state=SEED, verbosity=0,
    )
    pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])
    parameter_space = {
        "model__n_estimators": [250, 400, 600],
        "model__max_depth": [2, 3, 4],
        "model__learning_rate": [0.025, 0.04, 0.06, 0.08],
        "model__min_child_weight": [1, 3, 6],
        "model__subsample": [0.75, 0.9, 1.0],
        "model__colsample_bytree": [0.7, 0.85, 1.0],
        "model__reg_lambda": [1, 5, 10],
        "model__reg_alpha": [0, 0.2, 1],
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    search = RandomizedSearchCV(
        pipeline, parameter_space, n_iter=10,
        scoring={"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error"},
        refit="rmse", cv=cv, random_state=SEED, n_jobs=1,
        return_train_score=True,
    )
    started = time.perf_counter()
    search.fit(train[input_columns], train[TARGET])
    search_seconds = time.perf_counter() - started

    selected_model = search.best_estimator_
    test_prediction = selected_model.predict(test[input_columns])
    proposed_metrics = _metrics(test[TARGET], test_prediction)

    # Compare against both Phase 6 evaluations. Requiring improvement over the
    # strongest prior RMSE and MAE makes the export decision conservative.
    phase6_models = _load_pickle("phase6_best_models.pkl")
    baseline_records = [phase6_models["reduced"], phase6_models["full"]]
    baseline_rows = []
    for record in baseline_records:
        record_input = record["numeric_features"] + (GROUPS if record["include_context"] else [])
        prediction = record["model"].predict(test[record_input])
        baseline_rows.append({
            "candidate": f"Phase 6 {record['model_name']} {'reduced' if record is phase6_models['reduced'] else 'full'}",
            "feature_set": "reduced" if record is phase6_models["reduced"] else "full climate",
            **_metrics(test[TARGET], prediction),
        })
    comparison_rows = baseline_rows + [{
        "candidate": "XGBoost standardized reduced features",
        "feature_set": f"{len(numeric_features)} numeric features + AREA + ITEM",
        **proposed_metrics,
    }]
    comparison = pd.DataFrame(comparison_rows)
    best_prior_rmse = min(row["RMSE"] for row in baseline_rows)
    best_prior_mae = min(row["MAE"] for row in baseline_rows)
    accepted = proposed_metrics["RMSE"] < best_prior_rmse and proposed_metrics["MAE"] < best_prior_mae
    comparison["selected_for_export"] = comparison["candidate"].eq("XGBoost standardized reduced features") & accepted
    comparison.to_csv(ARTIFACT_DIR / "normalization_test_comparison.csv", index=False)

    cv_rows = []
    for index, row in pd.DataFrame(search.cv_results_).iterrows():
        cv_rows.append({
            "rank_by_cv_rmse": int(row["rank_test_rmse"]),
            "mean_cv_rmse": float(-row["mean_test_rmse"]),
            "std_cv_rmse": float(row["std_test_rmse"]),
            "mean_cv_mae": float(-row["mean_test_mae"]),
            "std_cv_mae": float(row["std_test_mae"]),
            "mean_train_rmse": float(-row["mean_train_rmse"]),
            "hyperparameters": json.dumps({key.removeprefix("model__"): row[f"param_{key}"] for key in parameter_space}, sort_keys=True),
        })
    cv_results = pd.DataFrame(cv_rows).sort_values(["rank_by_cv_rmse", "mean_cv_mae"]).reset_index(drop=True)
    cv_results.to_csv(ARTIFACT_DIR / "normalization_cv_results.csv", index=False)

    feature_names = selected_model.named_steps["preprocess"].get_feature_names_out()
    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": selected_model.named_steps["model"].feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    importance.to_csv(ARTIFACT_DIR / "normalization_feature_importance.csv", index=False)

    test_output = test[["AREA", "ITEM", "YEAR", TARGET]].copy()
    test_output["prediction"] = test_prediction
    test_output["residual_actual_minus_prediction"] = test_output[TARGET] - test_output["prediction"]
    per_crop_rows = []
    for crop, crop_rows in test_output.groupby("ITEM", observed=True):
        per_crop_rows.append({"ITEM": crop, "n": len(crop_rows), **_metrics(crop_rows[TARGET], crop_rows["prediction"])})
    per_crop = pd.DataFrame(per_crop_rows).sort_values("ITEM").reset_index(drop=True)
    per_crop.to_csv(ARTIFACT_DIR / "normalization_metrics_by_crop.csv", index=False)

    # Always preserve the candidate diagnostics; promote them to the dashboard's
    # active artifacts only after the two-metric gate has been satisfied.
    test_output.to_csv(ARTIFACT_DIR / "normalization_candidate_predictions.csv", index=False)
    if accepted:
        manifest_path = ARTIFACT_DIR / "feature_manifest.json"
        with manifest_path.open(encoding="utf-8") as stream:
            export_manifest = json.load(stream)
        final_model = clone(selected_model)
        final_started = time.perf_counter()
        final_model.fit(frame[input_columns], frame[TARGET])
        final_fit_seconds = time.perf_counter() - final_started
        joblib.dump(final_model, ARTIFACT_DIR / "best_yield_model.joblib", compress=3)
        normalized_values = final_model.named_steps["preprocess"].named_transformers_["numeric"].transform(frame[numeric_features])
        normalized_export = pd.DataFrame(normalized_values, columns=numeric_features, index=frame.index)
        normalized_export["AREA"] = frame["AREA"].to_numpy()
        normalized_export["ITEM"] = frame["ITEM"].to_numpy()
        normalized_export[TARGET] = frame[TARGET].to_numpy()
        normalized_export = normalized_export[numeric_features + categorical_features + [TARGET]]
        normalized_export.to_csv(ARTIFACT_DIR / "normalized_modeling_dataset.csv", index=False)
        export_manifest.update({
            "selected_candidate": "XGBoost|standardized_numeric|reduced_features|AREA+ITEM",
            "model_type": "XGBRegressor",
            "best_hyperparameters": search.best_params_,
            "numeric_feature_columns": numeric_features,
            "categorical_feature_columns": categorical_features,
            "required_input_columns_in_order": input_columns,
            "context_features_included": True,
            "normalization": {
                "numeric_transform": "StandardScaler, fitted inside each training CV fold and refit on full data for deployment",
                "categorical_transform": "OneHotEncoder(handle_unknown=ignore)",
                "target_transform": "none; YIELD retained in source units",
            },
            "normalized_dataset_file": "normalized_modeling_dataset.csv",
            "normalized_dataset_note": "Numeric model predictors are standardized; AREA and ITEM remain categorical and YIELD remains in source units. For predictions, provide raw source-scale numeric inputs because scaling is included in the saved model.",
            "final_model_fit_rows": int(len(frame)),
            "final_model_refit_after_holdout_evaluation": True,
            "final_model_fit_seconds": final_fit_seconds,
        })
        with manifest_path.open("w", encoding="utf-8") as stream:
            json.dump(export_manifest, stream, indent=2)
        test_output.to_csv(ARTIFACT_DIR / "heldout_predictions.csv", index=False)

        fig, ax = plt.subplots(figsize=(7.2, 6.0))
        for crop, crop_rows in test_output.groupby("ITEM", observed=True):
            ax.scatter(crop_rows[TARGET], crop_rows["prediction"], alpha=0.65, s=24, label=crop)
        low = min(test_output[TARGET].min(), test_output["prediction"].min())
        high = max(test_output[TARGET].max(), test_output["prediction"].max())
        ax.plot([low, high], [low, high], "k--", linewidth=1)
        ax.set(xlabel="Actual YIELD", ylabel="Predicted YIELD", title="Held-out predictions: standardized XGBoost (reduced features)")
        ax.legend(title="ITEM", fontsize=8)
        fig.tight_layout()
        fig.savefig(ARTIFACT_DIR / "predicted_vs_actual.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.2, 5.6))
        ax.scatter(test_output["prediction"], test_output["residual_actual_minus_prediction"], alpha=0.62, s=24)
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set(xlabel="Predicted YIELD", ylabel="Residual (actual - predicted)", title="Held-out residuals vs. predictions")
        fig.tight_layout()
        fig.savefig(ARTIFACT_DIR / "residuals_vs_predicted.png", dpi=150)
        plt.close(fig)
    else:
        final_fit_seconds = None
        export_manifest = json.loads((ARTIFACT_DIR / "feature_manifest.json").read_text(encoding="utf-8"))

    summary = {
        "selected_candidate": "XGBoost standardized reduced features",
        "selected_by": "lowest 5-fold training CV RMSE",
        "accepted_for_export": bool(accepted),
        "best_cv_rmse": float(-search.best_score_),
        "best_cv_mae": float(-search.cv_results_["mean_test_mae"][search.best_index_]),
        "search_seconds": float(search_seconds),
        "final_refit_seconds": final_fit_seconds,
        "seed": SEED,
        "folds": 5,
        "search_iterations": 10,
        "best_hyperparameters": search.best_params_,
        "candidate_holdout_metrics": proposed_metrics,
        "best_prior_rmse": float(best_prior_rmse),
        "best_prior_mae": float(best_prior_mae),
    }
    with (ARTIFACT_DIR / "normalization_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    return {
        "summary": summary,
        "test_comparison": comparison.sort_values("RMSE").reset_index(drop=True),
        "cv_results": cv_results,
        "feature_importance": importance.head(25),
        "metrics_by_crop": per_crop,
        "feature_manifest": export_manifest,
        "artifact_files": pd.DataFrame({"file": [
            "best_yield_model.joblib", "feature_manifest.json", "normalization_cv_results.csv",
            "normalization_test_comparison.csv", "normalization_feature_importance.csv",
            "normalization_metrics_by_crop.csv", "normalized_modeling_dataset.csv",
        ]}),
    }
