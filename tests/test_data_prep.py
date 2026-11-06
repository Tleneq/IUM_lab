"""Testy jednostkowe dla modułu data_prep (Zadanie 5).

Wymagane testy:
1. czy split_data zwraca zbiory o oczekiwanym rozmiarze (proporcja test_size),
2. czy suma rozmiarów zbioru treningowego i testowego równa się rozmiarowi całego zbioru,
3. czy funkcja zachowuje się przewidywalnie przy pustym lub niepoprawnym wejściu.
"""

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


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Fixture tworzący 100-wierszowy zbiór danych."""
    return pd.DataFrame({
        "feature_1": list(range(100)),
        "feature_2": [x * 2 for x in range(100)],
        "target": [0] * 50 + [1] * 50,
    })


# 1. Test proporcji test_size
def test_split_data_proportions(sample_df: pd.DataFrame):
    """Sprawdza, czy split_data zwraca zbiory o oczekiwanym rozmiarze (proporcja test_size)."""
    test_size = 0.25
    X_train, X_test, y_train, y_test = split_data(sample_df, test_size=test_size, seed=42)

    expected_test_len = int(len(sample_df) * test_size)
    assert len(X_test) == pytest.approx(expected_test_len, abs=1)
    assert len(y_test) == pytest.approx(expected_test_len, abs=1)
    assert "target" not in X_train.columns
    assert "target" not in X_test.columns


# 2. Test sumy rozmiarów zbiorów
def test_split_data_sum_of_sizes(sample_df: pd.DataFrame):
    """Sprawdza, czy suma rozmiarów zbioru treningowego i testowego równa się rozmiarowi całego zbioru."""
    total_len = len(sample_df)
    X_train, X_test, y_train, y_test = split_data(sample_df, test_size=0.2, seed=42)

    assert len(X_train) + len(X_test) == total_len
    assert len(y_train) + len(y_test) == total_len
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)


# 3. Test zachowania przy pustym lub niepoprawnym wejściu
def test_split_data_invalid_inputs():
    """Sprawdza czy funkcja zachowuje się przewidywalnie przy pustym lub niepoprawnym wejściu."""
    # Pusty DataFrame
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="nie może być pusta"):
        split_data(empty_df, test_size=0.2)

    # DataFrame o za małej liczbie wierszy (< 2)
    single_row_df = pd.DataFrame({"f1": [1], "target": [0]})
    with pytest.raises(ValueError, match="co najmniej 2 wiersze"):
        split_data(single_row_df, test_size=0.2)

    # Niepoprawny typ wejściowy zamiast DataFrame (np. None, lista, dict)
    with pytest.raises(TypeError, match="Oczekiwano typu pandas.DataFrame"):
        split_data(None, test_size=0.2)  # type: ignore

    with pytest.raises(TypeError, match="Oczekiwano typu pandas.DataFrame"):
        split_data([1, 2, 3], test_size=0.2)  # type: ignore

    # Niepoprawna wartość parametru test_size (ujemna, >= 1.0, 0.0)
    valid_df = pd.DataFrame({"f1": [1, 2, 3, 4], "target": [0, 1, 0, 1]})
    with pytest.raises(ValueError, match="musi być z przedziału"):
        split_data(valid_df, test_size=-0.2)

    with pytest.raises(ValueError, match="musi być z przedziału"):
        split_data(valid_df, test_size=1.5)

    with pytest.raises(ValueError, match="musi być z przedziału"):
        split_data(valid_df, test_size=0.0)

    # DataFrame bez kolumn cech (tylko kolumna target)
    only_target_df = pd.DataFrame({"target": [0, 1, 0, 1]})
    with pytest.raises(ValueError, match="co najmniej jedną kolumnę cech"):
        split_data(only_target_df, test_size=0.5, target_column="target")


# Dodatkowe testy integralności modułu
def test_load_data(sample_csv: Path):
    """Sprawdza wczytywanie pliku CSV do obiektu DataFrame."""
    df = load_data(sample_csv)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (10, 3)


def test_load_data_file_not_found():
    """Sprawdza rzucenie FileNotFoundError dla nieistniejącego pliku."""
    with pytest.raises(FileNotFoundError):
        load_data("non_existent_file_path_999.csv")


def test_split_data_reproducibility(sample_df: pd.DataFrame):
    """Sprawdza determinizm podziału przy tym samym ziarnie losowości."""
    X_train1, _, y_train1, _ = split_data(sample_df, test_size=0.2, seed=42)
    X_train2, _, y_train2, _ = split_data(sample_df, test_size=0.2, seed=42)
    X_train3, _, _, _ = split_data(sample_df, test_size=0.2, seed=123)

    pd.testing.assert_frame_equal(X_train1, X_train2)
    pd.testing.assert_series_equal(y_train1, y_train2)
    assert not X_train1.equals(X_train3)


def test_no_side_effects(sample_csv: Path, capsys):
    """Sprawdza czy funkcje nie generują efektów ubocznych w stdout/stderr."""
    df = load_data(sample_csv)
    _ = split_data(df, test_size=0.2, seed=42)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
