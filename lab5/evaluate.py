"""Moduł do ewaluacji zapisanego modelu scikit-learn (evaluate.py) dla Lab 5.

Wczytuje wcześniej wytrenowany model z katalogu models/ (.joblib) oraz dane testowe
z modułu data_prep.py, oblicza metryki (accuracy_score, classification_report, confusion_matrix)
i zapisuje podsumowanie do pliku (wyniki.txt).
"""

import os
import sys
from pathlib import Path
from typing import Any, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
for path_entry in (current_dir, parent_dir):
    if str(path_entry) not in sys.path:
        sys.path.insert(0, str(path_entry))

from data_prep import load_data, split_data

load_dotenv()

DEFAULT_DATA_PATH = os.getenv("DATA_PATH") or "data/wine.csv"
DEFAULT_TEST_SIZE = float(os.getenv("TEST_SIZE") or "0.2")
DEFAULT_RANDOM_SEED = int(os.getenv("RANDOM_SEED") or "42")
DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH") or "models/model.joblib"
DEFAULT_RESULTS_PATH = os.getenv("RESULTS_PATH") or "wyniki.txt"


def _resolve_model_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.exists():
        return p
    parent_p = Path("..") / path_str
    if parent_p.exists():
        return parent_p
    return p


def load_model(model_path: Union[str, Path, None] = None) -> Any:
    target = DEFAULT_MODEL_PATH if model_path is None else model_path
    file_path = _resolve_model_path(str(target))
    if not file_path.exists():
        raise FileNotFoundError(f"Plik modelu nie został znaleziony: {file_path}. Uruchom najpierw train.py.")
    return joblib.load(file_path)


def compute_metrics(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[float, str, np.ndarray]:
    y_pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    report = str(classification_report(y_test, y_pred))
    matrix = np.asarray(confusion_matrix(y_test, y_pred))
    return acc, report, matrix


def format_results(
    acc: float,
    report: str,
    matrix: np.ndarray,
    model_description: str = "Pipeline (StandardScaler -> Classifier)",
    test_samples: int = 0,
) -> str:
    lines = [
        "=" * 60,
        "EWALUACJA MODELU (evaluate.py - ZADANIE 3)",
        "=" * 60,
        f"Model: {model_description}",
        f"Liczba próbek testowych: {test_samples}",
        "-" * 60,
        f">>> ACCURACY SCORE: {acc:.4f} ({acc * 100:.2f}%) <<<",
        "-" * 60,
        "Classification Report:",
        report,
        "Confusion Matrix:",
        str(matrix),
        "=" * 60,
    ]
    return "\n".join(lines)


def save_results(result_text: str, results_path: Union[str, Path, None] = None) -> Path:
    target = DEFAULT_RESULTS_PATH if results_path is None else results_path
    file_path = Path(target)
    # Zapisz także do ../wyniki.txt jeśli istnieje katalog nadrzędny projektu
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(result_text + "\n")

    parent_path = Path("..") / target
    if Path("..").resolve() != file_path.resolve().parent and Path("../.git").exists():
        try:
            with open(parent_path, "w", encoding="utf-8") as pf:
                pf.write(result_text + "\n")
        except Exception:
            pass

    return file_path


def main() -> None:
    print("=" * 60)
    print("ETAP: EWALUACJA MODELU (evaluate.py)")
    print("=" * 60)
    print(f"Ścieżka modelu: {DEFAULT_MODEL_PATH}")
    print(f"Ścieżka danych: {DEFAULT_DATA_PATH}")
    print(f"Ścieżka zapisu wyników: {DEFAULT_RESULTS_PATH}")
    print("-" * 60)

    model = load_model(DEFAULT_MODEL_PATH)
    print("Pomyślnie załadowano model z dysku.")

    df = load_data(DEFAULT_DATA_PATH)
    _, X_test, _, y_test = split_data(
        df=df,
        test_size=DEFAULT_TEST_SIZE,
        seed=DEFAULT_RANDOM_SEED,
    )
    print(f"Wczytano zbiór testowy: {X_test.shape[0]} próbek, {X_test.shape[1]} cech")

    acc, report, matrix = compute_metrics(model, X_test, y_test)

    clf_name = "Model"
    if hasattr(model, "named_steps") and "classifier" in model.named_steps:
        clf_name = model.named_steps["classifier"].__class__.__name__

    report_text = format_results(
        acc=acc,
        report=report,
        matrix=matrix,
        model_description=f"StandardScaler -> {clf_name}",
        test_samples=len(y_test),
    )
    print("\n" + report_text)

    saved_file = save_results(report_text, DEFAULT_RESULTS_PATH)
    print(f"\nWyniki ewaluacji zapisano pomyślnie do: {saved_file.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
