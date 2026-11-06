"""Testy jednostkowe dla modułu evaluate (pytest)."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from evaluate import load_model, compute_metrics, save_results, format_results
from train import train_model


def test_load_model_file_not_found():
    """Sprawdza zgłaszanie FileNotFoundError dla nieistniejącego pliku modelu."""
    with pytest.raises(FileNotFoundError):
        load_model("non_existent_model_file_12345.joblib")


def test_compute_metrics():
    """Sprawdza obliczanie metryk jakości modelu na danych testowych."""
    X_train = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "f2": [10.0, 20.0, 30.0, 40.0]})
    y_train = pd.Series([0, 0, 1, 1])

    X_test = pd.DataFrame({"f1": [1.5, 3.5], "f2": [15.0, 35.0]})
    y_test = pd.Series([0, 1])

    model = train_model(X_train, y_train, model_type="logistic_regression", random_state=42)

    acc, report, matrix = compute_metrics(model, X_test, y_test)

    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0
    assert isinstance(report, str)
    assert len(report) > 0
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (2, 2)


def test_save_results(tmp_path: Path):
    """Sprawdza zapis raportu tekstowego do pliku."""
    result_text = "Przykładowy raport testowy"
    output_file = tmp_path / "wyniki_test.txt"

    saved_file = save_results(result_text, output_file)
    assert saved_file.exists()
    assert saved_file.read_text(encoding="utf-8").strip() == result_text
