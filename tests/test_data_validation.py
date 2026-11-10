from pathlib import Path

import pandas as pd
import pytest

from src.data import validate


def test_validate_file_passes_with_required_columns(tmp_path, monkeypatch):
    filename = "sample.csv"

    pd.DataFrame({
        "order_id": ["order-1"],
        "order_status": ["delivered"],
    }).to_csv(tmp_path / filename, index=False)

    monkeypatch.setattr(validate, "RAW_DATA_DIR", tmp_path)

    result = validate.validate_file(
        filename,
        ["order_id", "order_status"],
    )

    assert result is True


def test_validate_file_fails_when_column_is_missing(tmp_path, monkeypatch):
    filename = "sample.csv"

    pd.DataFrame({
        "order_id": ["order-1"],
    }).to_csv(tmp_path / filename, index=False)

    monkeypatch.setattr(validate, "RAW_DATA_DIR", tmp_path)

    result = validate.validate_file(
        filename,
        ["order_id", "order_status"],
    )

    assert result is False


def test_validate_file_raises_when_file_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(validate, "RAW_DATA_DIR", tmp_path)

    with pytest.raises(FileNotFoundError):
        validate.validate_file(
            "missing.csv",
            ["order_id"],
        )