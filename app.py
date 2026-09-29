"""Interactive Streamlit dashboard for the cereal yield prediction project."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "raw" / "europe_cereal_yield_climate_1990_2022.csv"
ARTIFACT_DIR = ROOT / "artifacts" / "cereal_yield_pipeline"
MANIFEST_PATH = ARTIFACT_DIR / "feature_manifest.json"
PREDICTIONS_PATH = ARTIFACT_DIR / "heldout_predictions.csv"

st.set_page_config(
    page_title="Cereal Yield Explorer",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp { background: #f5f7f4; }
      [data-testid="stHeader"] { background: rgba(245, 247, 244, 0.94); }
      .block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1500px; }
      .hero {
        padding: 1.6rem 1.8rem; border-radius: 20px; margin-bottom: 1.1rem;
        background: linear-gradient(115deg, #173b34 0%, #286b53 62%, #9aaf68 140%);
        color: #f7fbf5; box-shadow: 0 10px 30px rgba(26, 64, 48, .14);
      }
      .hero h1 { margin: 0 0 .35rem 0; font-size: 2.05rem; color: #f7fbf5; }
      .hero p { margin: 0; color: #dbe9dd; font-size: 1rem; }
      .eyebrow { color: #c7db9a; text-transform: uppercase; letter-spacing: .13em; font-size: .72rem; font-weight: 700; }
      [data-testid="stMetric"] {
        background: white; border: 1px solid #e4eae3; padding: .9rem 1rem;
        border-radius: 15px; box-shadow: 0 3px 12px rgba(31, 54, 42, .035);
      }
      div[data-testid="stForm"] { background: white; padding: 1rem 1.1rem; border: 1px solid #e4eae3; border-radius: 16px; }
      .small-note { color: #65766a; font-size: .88rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading the cereal dataset…")
def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


@st.cache_data(show_spinner=False)
def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Model manifest not found: {MANIFEST_PATH}. Run `python main.py` first."
        )
    with MANIFEST_PATH.open(encoding="utf-8") as stream:
        return json.load(stream)


@st.cache_resource(show_spinner="Loading the trained model…")
def load_model(model_path: str):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Trained model not found: {path}. Run `python main.py` first.")
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def load_predictions() -> pd.DataFrame:
    if not PREDICTIONS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(PREDICTIONS_PATH)


CLIMATE_LABELS = {
    "CDD": "Consecutive Dry Days(< 1mm/day)",
    "CSDI": "Cold Spell Duration Index",
    "CWD": "Consecutive Wet Days(≥ 1mm/day)",
    "HURS": "Relative Humidity(%)",
    "PR": "Precipitation(mm)",
    "R50MM": "Heavy Rain Days(≥ 50mm)",
    "RX5DAY": "Max 5-Day Rainfall(mm)",
    "TNN": "Coldest Night(°C)",
    "TR": "Tropical Nights(> 20°C)",
    "TR29": "Very Warm Nights(> 29°C)",
    "TR32": "Extremely Warm Nights(> 32°C)",
    "TX84RR": "Extreme Heat Recurrence",
    "TXX": "Hottest Day(°C)",
    "WSDI": "Warm Spell Duration Index",
}

DAY_COUNT_FEATURES = {"CDD", "CSDI", "CWD", "R50MM", "TR", "TR29", "TR32", "WSDI"}


def friendly_name(feature: str) -> str:
    code = feature.removeprefix("WB_CCKP_")
    return CLIMATE_LABELS.get(code, code.replace("_", " "))


def metric_summary(predictions: pd.DataFrame) -> dict[str, float] | None:
    required = {"YIELD", "prediction"}
    if predictions.empty or not required.issubset(predictions.columns):
        return None
    actual = predictions["YIELD"]
    predicted = predictions["prediction"]
    return {
        "r2": float(r2_score(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae": float(mean_absolute_error(actual, predicted)),
    }


try:
    data = load_dataset()
    manifest = load_manifest()
    model = load_model(str(ARTIFACT_DIR / manifest["model_file"]))
    heldout = load_predictions()
except Exception as error:
    st.error(str(error))
    st.stop()

areas = sorted(data["AREA"].dropna().astype(str).unique().tolist())
crops = sorted(data["ITEM"].dropna().astype(str).unique().tolist())
climate_features = manifest["climate_reduction"]["selected_climate_columns"]
numeric_features = manifest["numeric_feature_columns"]
required_columns = manifest["required_input_columns_in_order"]
year_min = int(data["YEAR"].min())
year_max = int(data["YEAR"].max())
scores = metric_summary(heldout)

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">European agriculture · 1990–2022</div>
      <h1>Cereal Yield Explorer</h1>
      <p>Explore observed yields, inspect the fitted model, and estimate yield from crop, country, year, and climate indicators.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### 🌾 Model card")
    st.caption(f"{manifest['model_type']} · seed {manifest['random_seed']}")
    st.caption(f"Trained on {manifest['final_model_fit_rows']:,} rows")
    st.divider()
    st.markdown("**Prediction inputs**")
    st.caption(f"{len(climate_features)} selected climate indicators + YEAR + AREA + ITEM")
    if manifest.get("normalization"):
        st.caption("Enter raw source-scale numeric values; the model standardizes them internally.")
    st.caption("Harvested area and production are excluded because they encode the target.")
    st.divider()
    st.markdown("**Run locally**")
    st.code("streamlit run app.py", language="bash")

overview_tab, predict_tab, insights_tab, explore_tab = st.tabs(
    ["Overview", "Predict yield", "Model insights", "Explore data"]
)


with overview_tab:
    st.subheader("At a glance")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Observations", f"{len(data):,}")
    col2.metric("Countries", f"{data['AREA'].nunique():,}")
    col3.metric("Crop labels", f"{data['ITEM'].nunique():,}")
    col4.metric("Climate features used", f"{len(climate_features)}")

    if scores:
        st.markdown("#### Held-out performance")
        score1, score2, score3 = st.columns(3)
        score1.metric("R²", f"{scores['r2']:.3f}")
        score2.metric("RMSE", f"{scores['rmse']:,.1f}", help="Lower is better; measured in the source YIELD units.")
        score3.metric("MAE", f"{scores['mae']:,.1f}", help="Lower is better; measured in the source YIELD units.")
    else:
        st.info("Run `python main.py` to create held-out predictions and evaluation artifacts.")

    left, right = st.columns([1, 1.15], gap="large")
    with left:
        st.markdown("#### Yield by crop")
        crop_stats = (
            data.groupby("ITEM", as_index=False)["YIELD"]
            .agg(mean_yield="mean", median_yield="median", observations="count")
            .sort_values("mean_yield", ascending=False)
        )
        st.bar_chart(crop_stats.set_index("ITEM")["mean_yield"], color="#3b8060")
        st.dataframe(crop_stats, hide_index=True, use_container_width=True)

    with right:
        st.markdown("#### Country yield levels")
        country_stats = (
            data.groupby("AREA", as_index=False)["YIELD"]
            .agg(mean_yield="mean", observations="count")
            .sort_values("mean_yield", ascending=False)
            .head(18)
        )
        st.bar_chart(country_stats.set_index("AREA")["mean_yield"], color="#879d50")
        st.caption("Showing the 18 countries with the highest observed mean yield.")

    st.markdown("#### Observed yield by year")
    trend_default = "France" if "France" in areas else areas[0]
    trend_country = st.selectbox("Country for annual view", areas, index=areas.index(trend_default))
    trend_rows = data.loc[data["AREA"].eq(trend_country)]
    annual = trend_rows.groupby(["YEAR", "ITEM"])["YIELD"].mean().unstack("ITEM")
    st.line_chart(annual, color=["#3b8060", "#c48b45", "#879d50"])

with predict_tab:
    st.subheader("Estimate a yield")
    st.write(
        "Choose a country and crop, then review or edit the climate values. "
        "Defaults come from the median observed values for that country/crop where available. "
        f"The model uses {year_max}, the latest year in the dataset."
    )

    selector1, selector2 = st.columns(2)
    with selector1:
        default_area = "France" if "France" in areas else areas[0]
        selected_area = st.selectbox("Country", areas, index=areas.index(default_area), key="prediction_area")
    with selector2:
        default_crop = "Wheat" if "Wheat" in crops else crops[0]
        selected_crop = st.selectbox("Crop", crops, index=crops.index(default_crop), key="prediction_crop")

    exact_group = data.loc[data["AREA"].eq(selected_area) & data["ITEM"].eq(selected_crop)]
    crop_group = data.loc[data["ITEM"].eq(selected_crop)]
    defaults_source = exact_group if not exact_group.empty else crop_group
    climate_defaults = defaults_source[climate_features].median(numeric_only=True)
    overall_defaults = data[climate_features].median(numeric_only=True)

    with st.form("yield_prediction_form"):
        st.markdown("##### Climate inputs")
        input_values: dict[str, float] = {}
        input_columns = st.columns(2)
        for index, feature in enumerate(climate_features):
            default_value = climate_defaults.get(feature, np.nan)
            if pd.isna(default_value):
                default_value = overall_defaults.get(feature, 0.0)
            default_value = float(default_value)
            is_day_count = feature.removeprefix("WB_CCKP_") in DAY_COUNT_FEATURES
            if is_day_count:
                default_value = float(np.floor(default_value + 0.5))
            step = 1.0 if is_day_count else max(abs(default_value) * 0.02, 0.01)
            with input_columns[index % 2]:
                input_values[feature] = st.number_input(
                    friendly_name(feature),
                    value=default_value,
                    step=float(step),
                    format="%.0f" if is_day_count else "%.4f",
                    key=(
                        f"input_{selected_area}_{selected_crop}_{feature}_whole_days"
                        if is_day_count
                        else f"input_{selected_area}_{selected_crop}_{feature}"
                    ),
                    help=(
                        f"Default is the rounded median from the selected group. Source feature: {feature}."
                        if is_day_count
                        else f"Default is the median from the selected group. Source feature: {feature}."
                    ),
                )
        submitted = st.form_submit_button("Predict yield", type="primary", use_container_width=True)

    if submitted:
        payload = {**input_values, "YEAR": int(year_max), "AREA": selected_area, "ITEM": selected_crop}
        prediction_frame = pd.DataFrame([payload], columns=required_columns)
        prediction = float(model.predict(prediction_frame)[0])
        st.success("Prediction complete")
        st.metric("Predicted yield (kg/ha)", f"{prediction:,.1f}")
        if not exact_group.empty:
            observed = exact_group["YIELD"]
            st.caption(
                f"Observed {selected_crop} yields for {selected_area}: median {observed.median():,.1f}; "
                f"range {observed.min():,.1f}–{observed.max():,.1f}. This is context, not a prediction interval."
            )
        with st.expander("Inspect exact model input"):
            st.dataframe(prediction_frame, hide_index=True, use_container_width=True)

    st.caption("Predictions are point estimates. The dashboard does not calculate prediction intervals.")


with insights_tab:
    st.subheader("Model diagnostics")
    if scores:
        metric_cols = st.columns(3)
        metric_cols[0].metric("Test R²", f"{scores['r2']:.3f}")
        metric_cols[1].metric("Test RMSE", f"{scores['rmse']:,.1f}")
        metric_cols[2].metric("Test MAE", f"{scores['mae']:,.1f}")

    if not heldout.empty and {"ITEM", "YIELD", "prediction"}.issubset(heldout.columns):
        crop_rows = []
        for crop, group in heldout.groupby("ITEM"):
            crop_rows.append({
                "ITEM": crop,
                "n": len(group),
                "R²": r2_score(group["YIELD"], group["prediction"]),
                "RMSE": float(np.sqrt(mean_squared_error(group["YIELD"], group["prediction"]))),
                "MAE": mean_absolute_error(group["YIELD"], group["prediction"]),
            })
        crop_metrics = pd.DataFrame(crop_rows).sort_values("ITEM")
        st.markdown("#### Test performance by crop")
        st.dataframe(crop_metrics, hide_index=True, use_container_width=True)
        st.bar_chart(crop_metrics.set_index("ITEM")["RMSE"], color="#c48b45")

        residuals = heldout.copy()
        residuals["residual"] = residuals["YIELD"] - residuals["prediction"]
        country_errors = (
            residuals.groupby("AREA", as_index=False)
            .agg(n=("residual", "size"), mean_residual=("residual", "mean"),
                 RMSE=("residual", lambda values: float(np.sqrt(np.mean(np.square(values))))))
            .sort_values("RMSE", ascending=False)
        )
        st.markdown("#### Country residual summary")
        st.dataframe(country_errors, hide_index=True, use_container_width=True, height=320)

    st.markdown("#### Feature importance")
    try:
        pipeline = model
        feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
        estimator = pipeline.named_steps["model"]
        if hasattr(estimator, "feature_importances_"):
            importances = pd.DataFrame({"feature": feature_names, "importance": estimator.feature_importances_})
            importances = importances.sort_values("importance", ascending=False).head(18)
            st.bar_chart(importances.set_index("feature")["importance"], color="#3b8060")
            item_total = float(
                pd.DataFrame({"feature": feature_names, "importance": estimator.feature_importances_})
                .loc[lambda frame: frame["feature"].str.contains("ITEM_", regex=False), "importance"].sum()
            )
            st.caption(f"Combined ITEM (crop) dummy importance: {item_total:.3f}.")
    except Exception as error:
        st.warning(f"Could not read model feature importance: {error}")

    plot_left, plot_right = st.columns(2)
    actual_plot = ARTIFACT_DIR / "predicted_vs_actual.png"
    residual_plot = ARTIFACT_DIR / "residuals_vs_predicted.png"
    with plot_left:
        st.markdown("#### Predicted vs. actual")
        if actual_plot.exists():
            st.image(str(actual_plot), use_container_width=True)
    with plot_right:
        st.markdown("#### Residuals vs. predicted")
        if residual_plot.exists():
            st.image(str(residual_plot), use_container_width=True)


with explore_tab:
    st.subheader("Explore observations")
    st.caption("The raw table includes accounting fields for audit purposes. They are not used by the prediction model.")
    filter1, filter2, filter3 = st.columns([1, 1.4, 1.5])
    with filter1:
        year_range = st.slider("Year range", year_min, year_max, (year_min, year_max))
    with filter2:
        selected_crops = st.multiselect("Crop labels", crops, default=crops)
    with filter3:
        selected_areas = st.multiselect("Countries", areas, default=areas)

    filtered = data.loc[
        data["YEAR"].between(year_range[0], year_range[1])
        & data["ITEM"].isin(selected_crops)
        & data["AREA"].isin(selected_areas)
    ].copy()
    st.write(f"Showing **{len(filtered):,}** of {len(data):,} observations.")
    if not filtered.empty:
        st.markdown("#### Yield observations over time")
        st.scatter_chart(
            filtered,
            x="YEAR",
            y="YIELD",
            color="ITEM",
            x_label="Year",
            y_label="Yield (kg/ha)",
            use_container_width=True,
        )
        st.dataframe(filtered, hide_index=True, use_container_width=True, height=520)
        st.download_button(
            "Download filtered CSV",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="filtered_cereal_yield_observations.csv",
            mime="text/csv",
        )
    else:
        st.info("No rows match the current filters.")

st.markdown(
    '<p class="small-note">Cross-sectional regression · YEAR is an ordinary numeric feature · no lag or rolling-window inputs</p>',
    unsafe_allow_html=True,
)
