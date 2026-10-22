"""Moduł do wczytywania i podziału wspólnego zbioru danych (Zadanie 1).

Wybór zbioru danych: Wine Recognition dataset (load_wine ze scikit-learn).
Zbiór jest dzielony funkcją train_test_split na część treningową (80%) i testową (20%)
ze stałym ziarnem losowości (random_state=42) oraz stratyfikacją klas.
Wyniki podziału zapisywane są do katalogu data/, aby kolejne modele korzystały
z dokładnie tych samych danych.
"""

import os
from pathlib import Path
from typing import Tuple, List
import pandas as pd
from dotenv import load_dotenv
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

# Załadowanie zmiennych środowiskowych
load_dotenv()

RANDOM_STATE = int(os.getenv("RANDOM_SEED", 42))
TEST_SIZE = float(os.getenv("TEST_SIZE", 0.2))
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))


def load_and_split_data(
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    save_to_disk: bool = True,
    data_dir: Path = DATA_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str], List[str]]:
    """Wczytuje zbiór Wine, wykonuje podział na zbiór treningowy i testowy

    oraz opcjonalnie zapisuje pliki do wskazanego katalogu.

    Returns:
        X_train: Cechy zbioru treningowego
        X_test: Cechy zbioru testowego
        y_train: Etykiety zbioru treningowego
        y_test: Etykiety zbioru testowego
        feature_names: Nazwy cech
        target_names: Nazwy klas
    """
    wine = load_wine(as_frame=True)
    X = wine.data
    y = wine.target
    feature_names = list(wine.feature_names)
    target_names = list(wine.target_names)

    # Podział 80/20 ze stałym random_state i stratyfikacją (zachowanie proporcji klas)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    if save_to_disk:
        data_dir.mkdir(parents=True, exist_ok=True)

        # Zapis pełnych ramek (z kolumną target)
        train_df = X_train.copy()
        train_df["target"] = y_train
        train_df.to_csv(data_dir / "train.csv", index=False)

        test_df = X_test.copy()
        test_df["target"] = y_test
        test_df.to_csv(data_dir / "test.csv", index=False)

        # Zapis pojedynczych składowych dla wygody
        X_train.to_csv(data_dir / "X_train.csv", index=False)
        X_test.to_csv(data_dir / "X_test.csv", index=False)
        y_train.to_csv(data_dir / "y_train.csv", index=False)
        y_test.to_csv(data_dir / "y_test.csv", index=False)

    return X_train, X_test, y_train, y_test, feature_names, target_names


def get_train_test_data(data_dir: Path = DATA_DIR):
    """Zwraca zbiory treningowe i testowe z plików CSV w data/.

    Jeśli pliki nie istnieją, generuje je automatycznie.
    """
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"

    if not train_path.exists() or not test_path.exists():
        return load_and_split_data(save_to_disk=True, data_dir=data_dir)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]
    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    wine = load_wine()
    return X_train, X_test, y_train, y_test, list(wine.feature_names), list(wine.target_names)


def main():
    print("=" * 60)
    print("ZADANIE 1: WSPÓLNY ZBIÓR DANYCH")
    print("=" * 60)
    print(f"Wybrany zbiór danych: Wine Recognition dataset (load_wine)")
    print(f"Konfiguracja: test_size={TEST_SIZE} (80% train / 20% test), random_state={RANDOM_STATE}")
    print("-" * 60)

    X_train, X_test, y_train, y_test, feature_names, target_names = load_and_split_data()

    total_samples = len(X_train) + len(X_test)
    num_features = len(feature_names)

    print(f"Liczba cech: {num_features}")
    print(f"Cechy: {', '.join(feature_names)}")
    print(f"Klasy: {', '.join(f'{i}: {name}' for i, name in enumerate(target_names))}")
    print("-" * 60)

    print(f"Całkowita liczba próbek: {total_samples}")
    print(f"Zbiór treningowy (X_train): {X_train.shape} | (y_train): {y_train.shape} ({len(X_train)/total_samples:.1%})")
    print(f"Zbiór testowy    (X_test):  {X_test.shape} | (y_test):  {y_test.shape} ({len(X_test)/total_samples:.1%})")
    print("-" * 60)

    print("Rozkład klas w zbiorze treningowym (y_train):")
    print(y_train.value_counts().sort_index().rename(index=dict(enumerate(target_names))))
    print("\nRozkład klas w zbiorze testowym (y_test):")
    print(y_test.value_counts().sort_index().rename(index=dict(enumerate(target_names))))
    print("-" * 60)

    print(f"Zapisano pliki danych do katalogu: {DATA_DIR.resolve()}")
    print(" - train.csv  (cechy + kolumna 'target')")
    print(" - test.csv   (cechy + kolumna 'target')")
    print(" - X_train.csv, y_train.csv")
    print(" - X_test.csv, y_test.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
