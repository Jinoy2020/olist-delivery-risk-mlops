from pathlib import Path
import sys
import json
from datetime import datetime, timezone
import argparse

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Ensure project root is available for imports
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import validate
from src.features import build_features, split_data



def create_run_directory():
    """Create a unique directory for pipeline execution reports."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    run_dir = PROJECT_ROOT / "reports" / "pipeline_runs" / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)

    return run_dir
def run_data_pipeline(
    overwrite=False,
    dry_run=False,
    train_only=False,
    overwrite_model=False,
):
    """Execute data validation, feature engineering and splitting."""

    print("\n========== OLIST MLOPS PIPELINE ==========\n")

    # Protect existing processed datasets
    protected_files = [
        PROJECT_ROOT / "data" / "processed" / "delivery_risk_features.csv",
        PROJECT_ROOT / "data" / "processed" / "train.csv",
        PROJECT_ROOT / "data" / "processed" / "validation.csv",
        PROJECT_ROOT / "data" / "processed" / "test.csv",
    ]

    existing_files = [path for path in protected_files if path.exists()]

    if train_only:
        required_files = [
            PROJECT_ROOT / "data" / "processed" / "train.csv",
            PROJECT_ROOT / "data" / "processed" / "validation.csv",
        ]

        missing_files = [
            path.name for path in required_files if not path.exists()
        ]

        if missing_files:
            raise FileNotFoundError(
                f"Required training files are missing: {', '.join(missing_files)}"
            )

        print("\n[TRAINING-ONLY MODE]")
        print("Using existing train.csv and validation.csv.")
        print("The test.csv file will not be used.")
    # Dry-run mode: preview without modifying files
    if dry_run:
        print("\n[DRY RUN] Pipeline execution preview")
        print("Existing processed files:")

        if existing_files:
            for path in existing_files:
                print(f"  - {path.name}")
        else:
            print("  None")

        if existing_files and not overwrite:
            print("\nExecution would be blocked to protect existing files.")
        else:
            print("\nPreflight passed. Pipeline stages are ready to execute.")

        return {
            "status": "dry_run",
            "existing_files": [path.name for path in existing_files],
            "overwrite": overwrite,
        }

    # Stop if existing files would be overwritten
    if existing_files and not overwrite and not train_only:
        filenames = ", ".join(path.name for path in existing_files)

        raise FileExistsError(
            f"Existing processed datasets detected: {filenames}. "
            "Use --overwrite to regenerate them intentionally."
        )

    run_dir = create_run_directory()

    results = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "stages": {},
        "status": "running",
    }

    try:
        if train_only:
            print("\n[TRAINING] Training model using existing splits...")

            from src.models import train

            model_output_dir = run_dir / "models"
            train.main(model_dir=model_output_dir)

           
            model_path = model_output_dir / "best_model.joblib"
            metrics_path = model_output_dir / "validation_metrics.json"

            if not model_path.exists() or not metrics_path.exists():
                raise RuntimeError(
                    "Training did not produce the expected model artifacts."
                )

            results["stages"]["model_training"] = {
                "status": "success",
                "model_path": str(model_path),
                "metrics_path": str(metrics_path),
            }
            with open(metrics_path, "r", encoding="utf-8") as file:
                validation_results = json.load(file)

            best_model_metrics = next(
                metric
                for metric in validation_results["validation_metrics"]
                if metric["model"] == validation_results["best_model"]
            )

            results["stages"]["model_evaluation"] = {
                "status": "success",
                "dataset": "validation",
                "best_model": validation_results["best_model"],
                "metrics": best_model_metrics,
            }

        else:
            print("\n[1/3] Validating raw datasets...")
            validate.main()
            results["stages"]["data_validation"] = "success"

            print("\n[2/3] Building features...")
            features_df = build_features.build_features()

            results["stages"]["feature_engineering"] = {
                "status": "success",
                "rows": len(features_df),
                "columns": len(features_df.columns),
            }

            print("\n[3/3] Creating chronological splits...")
            split_data.main()
            results["stages"]["data_splitting"] = "success"

        results["status"] = "success"
    except (Exception, SystemExit) as error:
        results["status"] = "failed"
        results["error"] = str(error)
        raise

    finally:
        results["completed_at"] = datetime.now(timezone.utc).isoformat()

        report_path = run_dir / "pipeline_run.json"

        with open(report_path, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=4)

        print(f"\nPipeline report: {report_path}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Olist MLOps data pipeline."
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow regeneration of existing processed datasets.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview pipeline execution without modifying datasets.",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Train a model using existing train and validation datasets.",
    )

    args = parser.parse_args()

    run_data_pipeline(
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        train_only=args.train_only,
    )