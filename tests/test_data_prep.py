"""Testy jednostkowe dla modułu data_prep (pytest)."""

from pathlib import Path
import pandas as pd
import pytest

from data_prep import load_data, split_data


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    """Fixture tworzący przykładowy plik CSV."""
    df = pd.DataFrame({
        "feature_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "feature_b": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        "target": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    })
    csv_file = tmp_path / "sample.csv"
    df.to_csv(csv_file, index=False)
    return csv_file


def test_load_data(sample_csv: Path):
    """Sprawdza czy load_data poprawnie wczytuje plik i zwraca DataFrame."""
    df = load_data(sample_csv)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (10, 3)
    assert list(df.columns) == ["feature_a", "feature_b", "target"]


def test_load_data_file_not_found():
    """Sprawdza czy podanie nieistniejącego pliku rzuca FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_data("non_existent_file_path_12345.csv")


def test_split_data_shapes():
    """Sprawdza poprawność podziału i proporcji w split_data."""
    df = pd.DataFrame({
        "f1": list(range(100)),
        "f2": list(range(100, 200)),
        "target": [0] * 50 + [1] * 50,
    })

    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, seed=42)

    assert len(X_train) == 80
    assert len(X_test) == 20
    assert len(y_train) == 80
    assert len(y_test) == 20
    assert "target" not in X_train.columns
    assert "target" not in X_test.columns
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)


def test_split_data_reproducibility():
    """Sprawdza determinizm podziału przy podaniu tego samego ziarna losowości."""
    df = pd.DataFrame({
        "x": list(range(50)),
        "target": [0, 1] * 25,
    })

    X_train1, X_test1, y_train1, y_test1 = split_data(df, test_size=0.3, seed=42)
    X_train2, X_test2, y_train2, y_test2 = split_data(df, test_size=0.3, seed=42)
    X_train3, _, _, _ = split_data(df, test_size=0.3, seed=99)

    pd.testing.assert_frame_equal(X_train1, X_train2)
    pd.testing.assert_series_equal(y_train1, y_train2)
    assert not X_train1.equals(X_train3)


def test_no_side_effects(sample_csv: Path, capsys):
    """Sprawdza czy funkcje działają bez efektów ubocznych (brak printów)."""
    df = load_data(sample_csv)
    _ = split_data(df, test_size=0.2, seed=42)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
