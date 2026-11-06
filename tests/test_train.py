"""Testy jednostkowe dla modułu train (pytest)."""

from pathlib import Path
import joblib
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from train import build_pipeline, train_model, save_model


def test_build_pipeline():
    """Sprawdza konstrukcję Pipeline dla regresji logistycznej oraz lasu losowego."""
    pipe_lr = build_pipeline(model_type="logistic_regression", random_state=42)
    assert isinstance(pipe_lr, Pipeline)
    assert "scaler" in pipe_lr.named_steps
    assert "classifier" in pipe_lr.named_steps
    assert pipe_lr.named_steps["classifier"].__class__.__name__ == "LogisticRegression"

    pipe_rf = build_pipeline(model_type="rf", random_state=42)
    assert pipe_rf.named_steps["classifier"].__class__.__name__ == "RandomForestClassifier"


def test_train_model():
    """Sprawdza czy funkcja train_model poprawnie trenuje Pipeline."""
    X_train = pd.DataFrame({
        "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "f2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    })
    y_train = pd.Series([0, 0, 0, 1, 1, 1])

    pipeline = train_model(X_train, y_train, model_type="logistic_regression", random_state=42)
    assert isinstance(pipeline, Pipeline)
    # Sprawdzenie czy model został dopasowany (posiada atrybut classes_)
    assert hasattr(pipeline.named_steps["classifier"], "classes_")


def test_save_model(tmp_path: Path):
    """Sprawdza zapisywanie modelu do pliku i możliwość jego odczytania."""
    pipeline = build_pipeline()
    model_file = tmp_path / "subdir" / "test_model.joblib"

    saved_path = save_model(pipeline, model_file)
    assert saved_path.exists()

    loaded = joblib.load(saved_path)
    assert isinstance(loaded, Pipeline)
