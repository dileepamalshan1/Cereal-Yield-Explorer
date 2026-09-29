# Europe cereal yield prediction

Cross-sectional regression for cereal yield from climate indicators, crop, country, and year. YEAR is an ordinary numeric feature; the pipeline does not create lags, rolling windows, or time-series forecasts.

## Run the pipeline

From the repository root, run:

```powershell
.\.venv\Scripts\python.exe main.py
```

This executes eight auditable phases and refreshes the trained model, feature manifest, held-out predictions, and plots in `artifacts/cereal_yield_pipeline/`. Phase 8 standardizes numeric predictors inside cross-validation, tunes XGBoost on the reduced feature set, and exports it only when it lowers both held-out RMSE and MAE. It also saves `normalized_modeling_dataset.csv`; YIELD stays in its original units, and the saved model expects raw source-scale inputs because it standardizes them internally. The phase notebooks and execution guide are in `notebooks/`.

## Run the Streamlit dashboard

Install the dashboard dependencies, then launch it from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The app has an overview of the observations and test metrics, a yield prediction form, model diagnostics and feature importance, and a filterable data explorer. Prediction defaults use the observed median climate values for the selected country/crop when available. Edit those inputs for a scenario estimate.

The trained model expects the ordered input columns in `artifacts/cereal_yield_pipeline/feature_manifest.json`. Harvested area and production quantity are intentionally excluded because they encode the target. Phase 8 saves its cross-validation results and held-out comparison alongside the model artifacts.
