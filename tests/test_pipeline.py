from src.pipeline.run_pipeline import run_data_pipeline
import pytest

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

    # Pipeline must stop before creating a run report.
    assert not (tmp_path / "reports" / "pipeline_runs").exists()

def test_dry_run_does_not_execute_pipeline_stages(monkeypatch):
    """Dry-run must not call validation, feature engineering or splitting."""

    import src.pipeline.run_pipeline as pipeline

    def unexpected_call():
        raise AssertionError("A pipeline stage was executed during dry-run.")

    monkeypatch.setattr(pipeline, "PROJECT_ROOT", __import__("pathlib").Path.cwd())
    monkeypatch.setattr(pipeline.validate, "main", unexpected_call)
    monkeypatch.setattr(pipeline.build_features, "build_features", unexpected_call)
    monkeypatch.setattr(pipeline.split_data, "main", unexpected_call)

    result = pipeline.run_data_pipeline(dry_run=True)

    assert result["status"] == "dry_run"
def test_training_only_uses_separate_model_directory(tmp_path, monkeypatch):
    """Training-only mode must train without rebuilding datasets."""

    import src.pipeline.run_pipeline as pipeline

    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)

    (processed_dir / "train.csv").touch()
    (processed_dir / "validation.csv").touch()

    monkeypatch.setattr(pipeline, "PROJECT_ROOT", tmp_path)

    captured = {}

    def fake_train(model_dir=None):
        captured["model_dir"] = model_dir

        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "best_model.joblib").touch()
        (model_dir / "validation_metrics.json").touch()

    def unexpected_call(*args, **kwargs):
        raise AssertionError("Data pipeline stage should not run.")

    import sys
    from types import ModuleType

    fake_train_module = ModuleType("src.models.train")
    fake_train_module.main = fake_train

    monkeypatch.setitem(
        sys.modules,
        "src.models.train",
        fake_train_module,
    )
    monkeypatch.setattr(pipeline.validate, "main", unexpected_call)
    monkeypatch.setattr(
        pipeline.build_features, "build_features", unexpected_call
    )
    monkeypatch.setattr(pipeline.split_data, "main", unexpected_call)

    result = pipeline.run_data_pipeline(train_only=True)

    assert result["status"] == "success"
    assert result["stages"]["model_training"]["status"] == "success"
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