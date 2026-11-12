
from src.pipeline.run_pipeline import run_data_pipeline
import pytest
from unittest.mock import Mock
import pandas as pd

def test_dry_run_protects_existing_files(tmp_path, monkeypatch):
    """Dry-run should detect existing datasets without executing stages."""

    import src.pipeline.run_pipeline as pipeline

    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)

    filenames = [
        "delivery_risk_features.csv",
        "train.csv",
        "validation.csv",
        "test.csv",
    ]

    for filename in filenames:
        (processed_dir / filename).touch()

    monkeypatch.setattr(pipeline, "PROJECT_ROOT", tmp_path)

    result = run_data_pipeline(dry_run=True)

    assert result["status"] == "dry_run"
    assert result["existing_files"] == filenames
    assert result["overwrite"] is False


def test_dry_run_with_no_existing_files(tmp_path, monkeypatch):
    """Dry-run should pass when no processed datasets exist."""

    import src.pipeline.run_pipeline as pipeline

    monkeypatch.setattr(pipeline, "PROJECT_ROOT", tmp_path)

    result = run_data_pipeline(dry_run=True)

    assert result["status"] == "dry_run"
    assert result["existing_files"] == []
    assert result["overwrite"] is False


def test_pipeline_blocks_overwrite_by_default(tmp_path, monkeypatch):
    """Normal execution must stop when processed files already exist."""

    import src.pipeline.run_pipeline as pipeline

    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)

    (processed_dir / "test.csv").touch()

    monkeypatch.setattr(pipeline, "PROJECT_ROOT", tmp_path)

    with pytest.raises(FileExistsError, match="Use --overwrite"):
        run_data_pipeline()

    assert not (tmp_path / "reports" / "pipeline_runs").exists()


def test_dry_run_does_not_execute_pipeline_stages(monkeypatch):
    """Dry-run must not execute pipeline stages."""

    import src.pipeline.run_pipeline as pipeline

    def unexpected_call():
        raise AssertionError("A pipeline stage was executed during dry-run.")

    monkeypatch.setattr(
        pipeline,
        "PROJECT_ROOT",
        __import__("pathlib").Path.cwd(),
    )
    monkeypatch.setattr(pipeline.validate, "main", unexpected_call)
    monkeypatch.setattr(
        pipeline.build_features,
        "build_features",
        unexpected_call,
    )
    monkeypatch.setattr(pipeline.split_data, "main", unexpected_call)

    result = pipeline.run_data_pipeline(dry_run=True)

    assert result["status"] == "dry_run"


def test_training_only_uses_separate_model_directory(tmp_path, monkeypatch):
    """Training-only mode must train without rebuilding datasets."""

    import src.pipeline.run_pipeline as pipeline
    from src.models import train as train_module

    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)

    (processed_dir / "train.csv").touch()
    (processed_dir / "validation.csv").touch()

    monkeypatch.setattr(pipeline, "PROJECT_ROOT", tmp_path)

    captured = {}

    def fake_train(model_dir=None, register_model=False):
        captured["model_dir"] = model_dir
        captured["register_model"] = register_model

        model_dir.mkdir(parents=True, exist_ok=True)

        (model_dir / "best_model.joblib").touch()

        (model_dir / "validation_metrics.json").write_text(
            """
            {
                "best_model": "logistic_regression",
                "validation_metrics": [
                    {
                        "model": "logistic_regression",
                        "pr_auc": 0.1377,
                        "roc_auc": 0.7502,
                        "precision": 0.1056,
                        "recall": 0.6765,
                        "f1": 0.1827
                    }
                ]
            }
            """,
            encoding="utf-8",
        )

        return {
            "best_model": "logistic_regression",
            "model_id": "test-model-id",
            "registered_model": None,
        }

    def unexpected_call(*args, **kwargs):
        raise AssertionError("Data pipeline stage should not run.")

    monkeypatch.setattr(train_module, "main", fake_train)
    monkeypatch.setattr(pipeline.validate, "main", unexpected_call)
    monkeypatch.setattr(
        pipeline.build_features,
        "build_features",
        unexpected_call,
    )
    monkeypatch.setattr(pipeline.split_data, "main", unexpected_call)

    result = pipeline.run_data_pipeline(train_only=True)

    assert captured["register_model"] is False

    assert result["status"] == "success"
    assert result["stages"]["model_training"]["status"] == "success"

    evaluation = result["stages"]["model_evaluation"]

    assert evaluation["status"] == "success"
    assert evaluation["dataset"] == "validation"
    assert evaluation["best_model"] == "logistic_regression"
    assert evaluation["metrics"]["model"] == "logistic_regression"
    assert abs(evaluation["metrics"]["pr_auc"] - 0.1377) < 0.0001

    assert captured["model_dir"].parent.parent == (
        tmp_path / "reports" / "pipeline_runs"
    )

    assert (captured["model_dir"] / "best_model.joblib").exists()


def test_training_only_fails_when_validation_data_is_missing(
    tmp_path, monkeypatch
):
    """Training-only mode must stop if a required split is missing."""

    import src.pipeline.run_pipeline as pipeline

    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)

    (processed_dir / "train.csv").touch()

    monkeypatch.setattr(pipeline, "PROJECT_ROOT", tmp_path)

    with pytest.raises(FileNotFoundError, match="validation.csv"):
        pipeline.run_data_pipeline(train_only=True)

    assert not (tmp_path / "reports" / "pipeline_runs").exists()


def test_register_best_model_calls_mlflow_registry(monkeypatch):
    """Registration must use the correct MLflow model URI."""

    from src.models import train

    mock_register = Mock(
        return_value=Mock(
            name="olist-delivery-risk-classifier",
            version="2",
        )
    )

    monkeypatch.setattr(
        train.mlflow,
        "register_model",
        mock_register,
    )

    result = train.register_best_model("test-model-id")

    mock_register.assert_called_once_with(
        model_uri="models:/test-model-id",
        name="olist-delivery-risk-classifier",
    )

    assert result.version == "2"


def test_training_only_records_model_registration(tmp_path, monkeypatch):
    """Training-only mode must record registration without real training."""

    import src.pipeline.run_pipeline as pipeline
    from src.models import train as train_module

    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)

    (processed_dir / "train.csv").touch()
    (processed_dir / "validation.csv").touch()

    monkeypatch.setattr(pipeline, "PROJECT_ROOT", tmp_path)

    captured = {}

    def fake_train(model_dir=None, register_model=False):
        captured["model_dir"] = model_dir
        captured["register_model"] = register_model

        model_dir.mkdir(parents=True, exist_ok=True)

        (model_dir / "best_model.joblib").touch()

        (model_dir / "validation_metrics.json").write_text(
            """
            {
                "best_model": "logistic_regression",
                "validation_metrics": [
                    {
                        "model": "logistic_regression",
                        "pr_auc": 0.1377,
                        "roc_auc": 0.7502,
                        "precision": 0.1056,
                        "recall": 0.6765,
                        "f1": 0.1827
                    }
                ]
            }
            """,
            encoding="utf-8",
        )

        return {
            "best_model": "logistic_regression",
            "model_id": "test-model-id",
            "registered_model": (
                {
                    "name": "olist-delivery-risk-classifier",
                    "version": "2",
                }
                if register_model
                else None
            ),
        }

    def unexpected_call(*args, **kwargs):
        raise AssertionError("Data pipeline stage should not run.")

    # Patch the actual imported training module.
    # This prevents real model training and MLflow registration.
    monkeypatch.setattr(train_module, "main", fake_train)

    monkeypatch.setattr(pipeline.validate, "main", unexpected_call)
    monkeypatch.setattr(
        pipeline.build_features,
        "build_features",
        unexpected_call,
    )
    monkeypatch.setattr(pipeline.split_data, "main", unexpected_call)

    result = pipeline.run_data_pipeline(
        train_only=True,
        register_model=True,
    )

    assert captured["register_model"] is True

    registration = result["stages"]["model_registration"]

    assert registration["status"] == "success"
    assert registration["model_name"] == "olist-delivery-risk-classifier"
    assert registration["version"] == "2"

    assert result["status"] == "success"
    assert result["stages"]["model_training"]["status"] == "success"

    evaluation = result["stages"]["model_evaluation"]

    assert evaluation["status"] == "success"
    assert evaluation["dataset"] == "validation"
    assert evaluation["best_model"] == "logistic_regression"
    assert evaluation["metrics"]["model"] == "logistic_regression"
    assert abs(evaluation["metrics"]["pr_auc"] - 0.1377) < 0.0001

    assert captured["model_dir"].parent.parent == (
        tmp_path / "reports" / "pipeline_runs"
    )

    assert (captured["model_dir"] / "best_model.joblib").exists()



def test_model_validation_passes():
    from src.models.train import validate_model_for_registration

    result = validate_model_for_registration(
        {
            "pr_auc": 0.50,
            "roc_auc": 0.75,
        },
        pd.Series([0, 0, 0, 0, 1, 1]),
    )

    assert result["passed"] is True
    assert result["reasons"] == []

def test_model_validation_fails_low_pr_auc():
    from src.models.train import validate_model_for_registration

    result = validate_model_for_registration(
        {
            "pr_auc": 0.20,
            "roc_auc": 0.75,
        },
        pd.Series([0, 0, 0, 1, 1]),
    )

    assert result["passed"] is False
    assert any(
        "PR-AUC" in reason
        for reason in result["reasons"]
    )


def test_model_validation_fails_low_roc_auc():
    from src.models.train import validate_model_for_registration

    result = validate_model_for_registration(
        {
            "pr_auc": 0.50,
            "roc_auc": 0.50,
        },
        pd.Series([0, 0, 0, 1, 1]),
    )

    assert result["passed"] is False
    assert any(
        "ROC-AUC" in reason
        for reason in result["reasons"]
    )


def test_model_validation_rejects_invalid_metrics():
    from src.models.train import validate_model_for_registration

    result = validate_model_for_registration(
        {
            "pr_auc": float("nan"),
            "roc_auc": 0.75,
        },
        pd.Series([0, 0, 0, 1, 1]),
    )

    assert result["passed"] is False


def test_failed_validation_blocks_model_registration(monkeypatch):
    from src.models import train

    mock_register = Mock()

    monkeypatch.setattr(
        train.mlflow,
        "register_model",
        mock_register,
    )

    with pytest.raises(ValueError, match="failed validation"):
        train.register_model_if_valid(
            model_id="test-model-id",
            metrics={
                "pr_auc": 0.05,
                "roc_auc": 0.45,
            },
            y_val=pd.Series([0, 0, 0, 0, 1, 1]),
        )

    mock_register.assert_not_called()


def test_validation_metadata_is_logged_to_selected_run(monkeypatch):
    from src.models import train

    logged = {}

    class FakeRun:
        def __init__(self, run_id):
            self.info = Mock(run_id=run_id)

        def __enter__(self):
            logged["run_id"] = self.info.run_id
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    monkeypatch.setattr(
        train.mlflow,
        "start_run",
        lambda run_id=None, **kwargs: FakeRun(run_id),
    )

    monkeypatch.setattr(
        train.mlflow,
        "log_params",
        lambda params: logged.update(params),
    )

    monkeypatch.setattr(
        train.mlflow,
        "log_metrics",
        lambda metrics: logged.update(metrics),
    )

    monkeypatch.setattr(
        train.mlflow,
        "set_tag",
        lambda key, value: logged.update({key: value}),
    )

    train.log_validation_metadata(
        "selected-run-123",
        {
            "passed": True,
            "baseline_pr_auc": 0.10,
        },
    )

    assert logged["run_id"] == "selected-run-123"
    assert logged["validation_gate_passed"] == "True"
    assert logged["validation_gate_rule"] == (
        "pr_auc_above_baseline_and_roc_auc_above_0.5"
    )
    assert logged["validation_baseline_pr_auc"] == 0.10
    assert logged["validation_gate_status"] == "passed"

def test_model_validation_handles_missing_and_non_numeric_metrics():
    from src.models.train import validate_model_for_registration

    y_val = pd.Series([0, 0, 0, 1, 1])

    missing_metric = validate_model_for_registration(
        {"pr_auc": 0.40},
        y_val,
    )

    non_numeric_metric = validate_model_for_registration(
        {
            "pr_auc": "invalid",
            "roc_auc": 0.75,
        },
        y_val,
    )

    assert missing_metric["passed"] is False
    assert missing_metric["roc_auc"] is None

    assert non_numeric_metric["passed"] is False
    assert non_numeric_metric["pr_auc"] is None