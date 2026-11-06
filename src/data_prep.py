"""Moduł do wczytywania i przygotowania danych (data_prep.py).

Zadanie 1 i Zadanie 4:
Separacja odpowiedzialności - moduł odpowiedzialny za wczytywanie i podział danych.
Wszystkie parametry (ścieżka, test_size, seed) wczytywane są z pliku .env za pomocą python-dotenv.
Funkcje load_data i split_data są czyste (brak efektów ubocznych, brak print/zapisu).
"""

import os
from pathlib import Path
from typing import Tuple, Union
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

# Wczytanie konfiguracji z pliku .env na poziomie modułu
load_dotenv()

DEFAULT_DATA_PATH = os.getenv("DATA_PATH", "data/wine.csv")
DEFAULT_TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
DEFAULT_RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
DEFAULT_DATA_DIR = Path(os.getenv("DATA_DIR", "data"))


def load_data(path: Union[str, Path, None] = None) -> pd.DataFrame:
    """Wczytuje zbiór danych ze wskazanej ścieżki i zwraca DataFrame.

    Funkcja nie posiada efektów ubocznych (brak printów i zapisu do pliku).

    Args:
        path: Ścieżka do pliku z danymi (np. CSV). Jeśli None, pobiera DATA_PATH z .env.

    Returns:
        pd.DataFrame: Wczytane dane.

    Raises:
        FileNotFoundError: Gdy wskazany plik nie istnieje.
        TypeError: Gdy wczytany obiekt nie jest ramką DataFrame.
    """
    if path is None:
        target_path = DEFAULT_DATA_PATH
    elif isinstance(path, Path):
        target_path = str(path)
    else:
        target_path = str(path)

    file_path = Path(target_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Plik danych nie został znaleziony: {file_path}")

    df = pd.read_csv(file_path)
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Oczekiwano typu pandas.DataFrame, otrzymano: {type(df)}")

    return df


def split_data(
    df: pd.DataFrame,
    test_size: Union[float, None] = None,
    seed: Union[int, None] = None,
    target_column: str = "target",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Dzieli DataFrame na zbiory uczący i testowy.

    Funkcja nie posiada efektów ubocznych (brak printów i zapisu do pliku).

    Args:
        df: Ramka danych zawierająca cechy i kolumnę docelową.
        test_size: Udział zbioru testowego (jeśli None, pobiera TEST_SIZE z .env).
        seed: Ziarno losowości (jeśli None, pobiera RANDOM_SEED z .env).
        target_column: Nazwa kolumny etykiety/celu (domyślnie 'target').
                       Jeśli kolumny nie ma w DataFrame, za cel przyjmuje się ostatnią kolumnę.

    Returns:
        Krotka (X_train, X_test, y_train, y_test).

    Raises:
        TypeError: Gdy df nie jest obiektem DataFrame.
        ValueError: Gdy df jest puste, ma za mało wierszy, brak cech lub test_size jest poza (0, 1).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Oczekiwano typu pandas.DataFrame, otrzymano: {type(df)}")

    if df.empty:
        raise ValueError("Ramka danych df nie może być pusta.")

    if len(df) < 2:
        raise ValueError("Ramka danych df musi zawierać co najmniej 2 wiersze do podziału.")

    actual_test_size = DEFAULT_TEST_SIZE if test_size is None else float(test_size)
    if not (0.0 < actual_test_size < 1.0):
        raise ValueError(
            f"Parametr test_size musi być z przedziału (0, 1), otrzymano: {actual_test_size}"
        )

    actual_seed = DEFAULT_RANDOM_SEED if seed is None else int(seed)

    if target_column in df.columns:
        X = df.drop(columns=[target_column])
        y = df[target_column]
    else:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

    if X.shape[1] == 0:
        raise ValueError("Ramka danych musi zawierać co najmniej jedną kolumnę cech (X).")

    # Bezpieczna stratyfikacja (jeśli każda klasa ma co najmniej 2 instancje)
    stratify = None
    if hasattr(y, "value_counts"):
        counts = y.value_counts()
        if len(counts) > 1 and (counts >= 2).all():
            stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=actual_test_size,
        random_state=actual_seed,
        stratify=stratify,
    )

    return X_train, X_test, y_train, y_test


def main() -> None:
    """Główna funkcja CLI do przygotowania danych w potoku CI/CD.

    Wczytuje parametry z pliku .env, przetwarza dane i zapisuje zbiory do katalogu data/.
    """
    print("=" * 60)
    print("ETAP: PRZYGOTOWANIE DANYCH (data_prep.py)")
    print("=" * 60)
    print(f"Ścieżka danych wejściowych: {DEFAULT_DATA_PATH}")
    print(f"Parametry: test_size={DEFAULT_TEST_SIZE}, random_seed={DEFAULT_RANDOM_SEED}")
    print("-" * 60)

    # 1. Wczytanie danych
    df = load_data(DEFAULT_DATA_PATH)
    print(f"Wczytano DataFrame o wymiarach: {df.shape}")

    # 2. Podział danych
    X_train, X_test, y_train, y_test = split_data(
        df=df,
        test_size=DEFAULT_TEST_SIZE,
        seed=DEFAULT_RANDOM_SEED,
    )

    # 3. Zapis plików dla kolejnych etapów potoku (train.py / evaluate.py)
    DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_df = X_train.copy()
    train_df["target"] = y_train
    train_df.to_csv(DEFAULT_DATA_DIR / "train.csv", index=False)

    test_df = X_test.copy()
    test_df["target"] = y_test
    test_df.to_csv(DEFAULT_DATA_DIR / "test.csv", index=False)

    X_train.to_csv(DEFAULT_DATA_DIR / "X_train.csv", index=False)
    X_test.to_csv(DEFAULT_DATA_DIR / "X_test.csv", index=False)
    y_train.to_csv(DEFAULT_DATA_DIR / "y_train.csv", index=False)
    y_test.to_csv(DEFAULT_DATA_DIR / "y_test.csv", index=False)

    total = len(df)
    print(f"Zbiór treningowy: {X_train.shape[0]} próbek ({X_train.shape[0] / total:.1%})")
    print(f"Zbiór testowy:    {X_test.shape[0]} próbek ({X_test.shape[0] / total:.1%})")
    print(f"Zapisano pliki do katalogu: {DEFAULT_DATA_DIR.resolve()}")
    print(" - train.csv, test.csv")
    print(" - X_train.csv, y_train.csv, X_test.csv, y_test.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
