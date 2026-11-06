"""Moduł do trenowania modelu scikit-learn (train.py).

Zadanie 2:
Wczytuje i dzieli dane za pomocą funkcji load_data i split_data z data_prep.py,
tworzy potok (Pipeline) ze skalowaniem StandardScaler i klasyfikatorem,
trenuje model na zbiorze treningowym i zapisuje go w katalogu models/ w formacie .joblib.
"""

import os
import sys
from pathlib import Path
from typing import Any, Union
import joblib
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Zapewnienie dostępu do data_prep niezależnie od katalogu wywołania
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
for path_entry in (current_dir, parent_dir):
    if str(path_entry) not in sys.path:
        sys.path.insert(0, str(path_entry))

from data_prep import load_data, split_data


def build_pipeline(
    model_type: str = "logistic_regression",
    random_state: int = 42,
) -> Pipeline:
    """Tworzy pipeline łączący StandardScaler z wybranym klasyfikatorem.

    Args:
        model_type: Typ modelu ('logistic_regression' lub 'random_forest' / 'rf').
        random_state: Ziarno losowości.

    Returns:
        Skonstruowany obiekt sklearn.pipeline.Pipeline.
    """
    normalized_type = model_type.lower()
    if normalized_type in {"rf", "random_forest", "randomforestclassifier"}:
        classifier = RandomForestClassifier(random_state=random_state)
    else:
        classifier = LogisticRegression(random_state=random_state, max_iter=1000)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", classifier),
    ])
    return pipeline


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "logistic_regression",
    random_state: int = 42,
) -> Pipeline:
    """Tworzy i trenuje pipeline na dostarczonych danych treningowych.

    Args:
        X_train: Cechy zbioru treningowego.
        y_train: Etykiety zbioru treningowego.
        model_type: Nazwa algorytmu klasyfikatora.
        random_state: Ziarno losowości.

    Returns:
        Wytrenowany obiekt Pipeline.
    """
    pipeline = build_pipeline(model_type=model_type, random_state=random_state)
    pipeline.fit(X_train, y_train)
    return pipeline


def save_model(model: Any, model_path: Union[str, Path]) -> Path:
    """Zapisuje wytrenowany model do pliku za pomocą biblioteki joblib.

    Args:
        model: Obiekt modelu lub pipeline do zapisu.
        model_path: Ścieżka docelowa pliku (.joblib).

    Returns:
        Path: Bezwzględna ścieżka do zapisanego modelu.
    """
    file_path = model_path if isinstance(model_path, Path) else Path(model_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, file_path)
    return file_path


def main() -> None:
    """Główna funkcja wykonująca pełny proces trenowania i zapisu modelu."""
    load_dotenv()

    data_path = os.getenv("DATA_PATH", "data/wine.csv")
    test_size = float(os.getenv("TEST_SIZE", "0.2"))
    random_seed = int(os.getenv("RANDOM_SEED", "42"))
    model_type = os.getenv("MODEL_TYPE", "logistic_regression")
    model_path = os.getenv("MODEL_PATH", "models/model.joblib")

    print("=" * 60)
    print("ETAP: TRENOWANIE MODELU (train.py)")
    print("=" * 60)
    print(f"Dane wejściowe: {data_path}")
    print(f"Typ modelu: {model_type} (seed={random_seed})")
    print(f"Docelowa ścieżka modelu: {model_path}")
    print("-" * 60)

    # 1. Wczytanie i podział danych za pomocą data_prep
    df = load_data(data_path)
    X_train, X_test, y_train, y_test = split_data(
        df=df,
        test_size=test_size,
        seed=random_seed,
    )
    print(f"Rozmiar zbioru treningowego: {X_train.shape[0]} próbek, {X_train.shape[1]} cech")

    # 2. Trening potoku Pipeline (StandardScaler -> Classifier)
    pipeline = train_model(
        X_train=X_train,
        y_train=y_train,
        model_type=model_type,
        random_state=random_seed,
    )
    clf_name = pipeline.named_steps["classifier"].__class__.__name__
    print(f"Wytrenowano Pipeline: StandardScaler -> {clf_name}")

    # 3. Zapis modelu do katalogu models/
    saved_path = save_model(pipeline, model_path)
    print(f"Model pomyślnie zapisany do: {saved_path.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
