"""Moduł do trenowania, ewaluacji i walidacji krzyżowej modelu scikit-learn (Zadania 2 i 4).

Wykorzystuje obiekt sklearn.pipeline.Pipeline łączący StandardScaler
oraz klasyfikator (LogisticRegression / RandomForestClassifier).
- Zadanie 2: Trenuje model metodą fit() na zbiorze treningowym i ocenia na zbiorze testowym.
- Zadanie 4: Uruchamia 5-krotną walidację krzyżową (KFold z shuffle=True) na zbiorze treningowym,
  oblicza wynik dla każdego foldu, średnią dokładność oraz odchylenie standardowe.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from load_dataset import get_train_test_data

# Załadowanie konfiguracji ze środowiska
load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))
RESULTS_PATH = Path(os.getenv("RESULTS_PATH", "wyniki.txt"))
MODEL_TYPE = os.getenv("MODEL_TYPE", "logistic_regression").lower()
CV_FOLDS = int(os.getenv("CV_FOLDS", 5))


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


def run_cross_validation(pipeline: Pipeline, X, y, n_splits: int = CV_FOLDS, random_state: int = RANDOM_SEED):
    """Przeprowadza n-krotną walidację krzyżową (KFold shuffle=True) dla danego potoku."""
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(pipeline, X, y, cv=kfold, scoring="accuracy")
    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))
    return scores, mean_score, std_score, kfold


def append_or_update_results(section_header: str, section_content: str, file_path: Path):
    """Pomocnicza funkcja do aktualizowania lub dopisywania sekcji w pliku wyników."""
    existing_content = ""
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            existing_content = f.read().strip()

    if section_header in existing_content:
        # Podmiana istniejącej sekcji
        parts = existing_content.split(section_header)
        prefix = parts[0].strip()
        # Sprawdzamy czy za sekcją jest kolejna sekcja z linią podziału
        rest = parts[1]
        next_sep = "\n============================================================"
        if next_sep in rest:
            suffix = rest[rest.index(next_sep):]
            new_text = f"{prefix}\n\n{section_header}\n{section_content.strip()}{suffix}\n"
        else:
            new_text = f"{prefix}\n\n{section_header}\n{section_content.strip()}\n"
    else:
        if existing_content:
            new_text = f"{existing_content}\n\n{section_header}\n{section_content.strip()}\n"
        else:
            new_text = f"{section_header}\n{section_content.strip()}\n"

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(new_text)


def main():
    print("=" * 60)
    print("ZADANIA 2 & 4: MODEL W SCIKIT-LEARN (PIPELINE & K-FOLD CV)")
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

    # --- Zadanie 2: Ewaluacja na zbiorze testowym ---
    acc, report, matrix, y_pred = evaluate_model(
        pipeline, X_train, y_train, X_test, y_test, target_names=target_names
    )

    clf_name = pipeline.named_steps["classifier"].__class__.__name__

    output_lines_task2 = [
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
    print("\n".join(output_lines_task2))

    # --- Zadanie 4: 5-krotna walidacja krzyżowa (KFold) ---
    print("\n" + "=" * 60)
    print("ZADANIE 4: 5-KROTNA WALIDACJA KRZYŻOWA (K-FOLD CV)")
    print("=" * 60)

    cv_scores, cv_mean, cv_std, kfold = run_cross_validation(
        pipeline, X_train, y_train, n_splits=CV_FOLDS, random_state=RANDOM_SEED
    )

    fold_details = []
    for i, score in enumerate(cv_scores, start=1):
        fold_line = f"  Fold {i}: Accuracy = {score:.4f} ({score * 100:.2f}%)"
        fold_details.append(fold_line)
        print(fold_line)

    print("-" * 60)
    print(f">>> ŚREDNIA DOKŁADNOŚĆ (Mean CV Accuracy): {cv_mean:.4f} ({cv_mean * 100:.2f}%) <<<")
    print(f">>> ODCHYLENIE STANDARDOWE (Std Dev):    {cv_std:.4f} (±{cv_std * 100:.2f}%) <<<")
    print("=" * 60)

    task4_content = (
        f"Pipeline: StandardScaler -> {clf_name}\n"
        f"Metoda: KFold (n_splits={CV_FOLDS}, shuffle=True, random_state={RANDOM_SEED})\n"
        f"Zbiór: Treningowy z Zadania 1 ({len(y_train)} próbek)\n"
        f"------------------------------------------------------------\n"
        f"Wyniki dla poszczególnych foldów:\n"
        + "\n".join(fold_details)
        + f"\n------------------------------------------------------------\n"
        f">>> ŚREDNIA DOKŁADNOŚĆ (Mean CV Accuracy): {cv_mean:.4f} ({cv_mean * 100:.2f}%) <<<\n"
        f">>> ODCHYLENIE STANDARDOWE (Std Dev):    {cv_std:.4f} (±{cv_std * 100:.2f}%) <<<\n"
        f"============================================================"
    )

    header_task4 = (
        "============================================================\n"
        "WALIDACJA KRZYŻOWA SCIKIT-LEARN (ZADANIE 4)"
    )

    append_or_update_results(header_task4, task4_content, RESULTS_PATH)
    print(f"\nWyniki walidacji krzyżowej zapisane do: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
