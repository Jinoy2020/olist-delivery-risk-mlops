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