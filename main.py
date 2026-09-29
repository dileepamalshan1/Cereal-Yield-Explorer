"""Run the complete cross-sectional cereal yield pipeline."""

from __future__ import annotations

import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
# Keep Matplotlib's font cache inside the writable project artifacts directory.
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(ROOT / "artifacts" / "cereal_yield_pipeline" / ".matplotlib"),
)

from notebooks.pipeline_helpers import (  # noqa: E402
    ARTIFACT_DIR,
    SEED,
    phase1,
    phase2,
    phase3,
    phase4,
    phase5,
    phase6,
    phase7,
    phase8,
)


def show(title: str, value) -> None:
    print(f"\n{title}")
    if hasattr(value, "to_string"):
        print(value.to_string(index=False, max_rows=200))
    else:
        print(value)


def main() -> None:
    started = time.perf_counter()
    print(f"Cereal yield pipeline | seed={SEED} | artifacts={ARTIFACT_DIR}")

    print("\n=== Phase 1: cleaning and exploration ===")
    p1 = phase1()
    for key in ["shape", "memory_summary", "duplicate_table", "domain_table", "cleaned_shape", "overall_yield", "crop_yield"]:
        show(key, p1[key])
    print(f"Outlier-flagged measure observations retained: {len(p1['outlier_table'])}")
    show("Year coverage gaps", p1["coverage_table"].loc[p1["coverage_table"]["missing_year_count"] > 0])

    print("\n=== Phase 2: leakage check ===")
    p2 = phase2()
    show("Production/area comparison", p2["summary"])
    show("Dropped leakage columns", p2["leakage_columns_dropped"])

    print("\n=== Phase 3: feature reduction and VIF ===")
    p3 = phase3()
    show("Low-variance variables dropped", p3["low_variance"])
    show("Correlation clusters and retained representatives", p3["cluster_report"])
    show("Selected climate features", p3["selected_climate"])
    show("VIF", p3["vif"])

    print("\n=== Phase 4: preprocessing and split ===")
    p4 = phase4()
    show("Train/test counts", p4["counts"])
    show("Split and preprocessing choices", p4["choice_table"])

    print("\n=== Phase 5: tuned candidate models ===")
    p5 = phase5()
    show("Training CV results, hyperparameters, and search time", p5["cv_table"])

    print("\n=== Phase 6: holdout evaluation ===")
    p6 = phase6()
    show("Candidate holdout comparison", p6["candidate_test"])
    show("Full versus reduced climate features", p6["reduction_comparison"])
    show("Top model features", p6["feature_importance"])
    show("Crop feature-group importance", p6["item_importance"])
    show("Metrics by crop", p6["per_crop"])
    show("Largest country residual RMSE", p6["per_country_worst"])
    print(f"Diagnostic plots: {p6['prediction_plot_path']} | {p6['residual_plot_path']}")

    print("\n=== Phase 7: final refit and export ===")
    p7 = phase7()
    show("Exported files", p7["artifact_files"])
    show("Required input columns in order", p7["feature_manifest"]["required_input_columns_in_order"])
    print(f"Final model refit rows: {p7['feature_manifest']['final_model_fit_rows']}")
    print(f"Final model fit time: {p7['final_fit_seconds']:.2f} seconds")

    print("\n=== Phase 8: normalized model optimization ===")
    p8 = phase8()
    show("Training-only CV search results", p8["cv_results"])
    show("Held-out comparison (RMSE and MAE in original YIELD units)", p8["test_comparison"])
    show("Best model feature importance", p8["feature_importance"])
    show("Held-out metrics by crop", p8["metrics_by_crop"])
    show("Optimization summary", p8["summary"])
    if p8["summary"]["accepted_for_export"]:
        print("Normalized XGBoost improved both RMSE and MAE and is now the exported model.")
    else:
        print("No candidate improved both error metrics; the prior exported model remains active.")
    print(f"Total pipeline runtime: {(time.perf_counter() - started) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
