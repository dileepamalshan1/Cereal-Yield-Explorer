# Cereal yield prediction notebooks

Run the notebooks in order from the repository root. The phase notebooks save intermediate tables and fitted estimators under `artifacts/cereal_yield_pipeline/`.

To run the full pipeline from the command line, use `python main.py` from the repository root (or `.venv/Scripts/python.exe main.py` on Windows). It runs every phase, prints the main audit and evaluation tables, and refreshes the exported model and diagnostics.

1. `00-ingest.ipynb` — dataset and workflow overview
2. `01_phase1_cleaning_exploration.ipynb` — data quality, outliers, yield summaries, and year coverage
3. `02_phase2_leakage.ipynb` — production/area arithmetic leakage check
4. `03_phase3_feature_reduction.ipynb` — training-only variance/correlation reduction and VIF
5. `04_phase4_preprocessing_split.ipynb` — fixed stratified split and preprocessing choices
6. `05_phase5_model_training.ipynb` — training-only cross-validated tuning
7. `06_phase6_evaluation.ipynb` — holdout results, full/reduced comparison, importances, and diagnostics
8. `07_phase7_export.ipynb` — model and feature-contract export

9. `08_normalization_optimization.ipynb` - predictor standardization, reduced-feature XGBoost tuning, and error comparison

The workflow uses seed 42, a shuffled 80/20 crop-stratified holdout, and shuffled five-fold CV. It treats YEAR as a regular numeric feature and does not create time-series features. The shared implementation is in `pipeline_helpers.py`; the notebooks call one phase at a time so the intermediate output remains reviewable.

The final exported model is refit on all cleaned observations after the held-out comparison. The performance tables and saved prediction CSV remain based on the untouched test split.

The environment needs pandas, NumPy, SciPy, scikit-learn, matplotlib, joblib, and XGBoost. The model artifact and its ordered prediction input contract are `artifacts/cereal_yield_pipeline/best_yield_model.joblib` and `artifacts/cereal_yield_pipeline/feature_manifest.json`.
