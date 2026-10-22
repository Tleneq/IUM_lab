"""Moduł do trenowania i ewaluacji modelu scikit-learn (Zadanie 2).

Wykorzystuje obiekt sklearn.pipeline.Pipeline łączący StandardScaler
oraz klasyfikator (LogisticRegression / RandomForestClassifier).
Trenuje model metodą fit() na zbiorze treningowym wygenerowanym w Zadaniu 1,
następnie wykonuje predykcję predict() na zbiorze testowym oraz oblicza
i wypisuje accuracy_score jako punkt odniesienia do dalszego porównania.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from load_dataset import get_train_test_data

# Załadowanie konfiguracji ze środowiska
load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))
RESULTS_PATH = Path(os.getenv("RESULTS_PATH", "wyniki.txt"))
MODEL_TYPE = os.getenv("MODEL_TYPE", "logistic_regression").lower()


def build_pipeline(model_type: str = "logistic_regression", random_state: int = RANDOM_SEED) -> Pipeline:
    """Tworzy pipeline łączący skalowanie StandardScaler z wybranym klasyfikatorem."""
    if model_type in ["rf", "random_forest", "randomforestclassifier"]:
        classifier = RandomForestClassifier(random_state=random_state)
    else:
        # Domyślnie LogisticRegression
        classifier = LogisticRegression(random_state=random_state, max_iter=1000)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", classifier),
    ])
    return pipeline


def evaluate_model(pipeline: Pipeline, X_train, y_train, X_test, y_test, target_names=None):
    """Trenuje pipeline metodą fit() i ewaluuje na zbiorze testowym za pomocą predict()."""
    # 1. Trenowanie potoku (Pipeline.fit)
    pipeline.fit(X_train, y_train)

    # 2. Predykcja na zbiorze testowym (Pipeline.predict)
    y_pred = pipeline.predict(X_test)

    # 3. Obliczenie accuracy_score jako punktu odniesienia
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names)
    matrix = confusion_matrix(y_test, y_pred)

    return acc, report, matrix, y_pred


def main():
    print("=" * 60)
    print("ZADANIE 2: MODEL W SCIKIT-LEARN (PIPELINE)")
    print("=" * 60)

    # Wczytanie wspólnego podziału danych z Zadania 1
    X_train, X_test, y_train, y_test, feature_names, target_names = get_train_test_data(data_dir=DATA_DIR)

    print(f"Dane wczytane z katalogu: {DATA_DIR}")
    print(f" - X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f" - X_test:  {X_test.shape}, y_test:  {y_test.shape}")
    print(f" - Wybrany model: {MODEL_TYPE}")
    print("-" * 60)

    # Budowa i trening pipeline'u scikit-learn
    pipeline = build_pipeline(model_type=MODEL_TYPE, random_state=RANDOM_SEED)
    print(f"Zbudowany Pipeline:\n{pipeline}\n")

    acc, report, matrix, y_pred = evaluate_model(
        pipeline, X_train, y_train, X_test, y_test, target_names=target_names
    )

    clf_name = pipeline.named_steps["classifier"].__class__.__name__

    output_lines = [
        "=" * 60,
        "EWALUACJA MODELU SCIKIT-LEARN (BASELINE - ZADANIE 2)",
        "=" * 60,
        f"Pipeline: StandardScaler -> {clf_name}",
        f"Parametry: RANDOM_SEED={RANDOM_SEED}",
        f"Rozmiar zbioru testowego: {len(y_test)} próbek",
        "-" * 60,
        f">>> ACCURACY SCORE (punkt odniesienia): {acc:.4f} ({acc * 100:.2f}%) <<<",
        "-" * 60,
        "Classification Report:",
        report,
        "Confusion Matrix:",
        str(matrix),
        "=" * 60,
    ]

    result_text = "\n".join(output_lines)
    print(result_text)

    # Zapis wyników do pliku
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(result_text + "\n")

    print(f"\nWynik punktu odniesienia zapisany pomyślnie do: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
