"""Moduł do wczytywania i przygotowania danych (data_prep.py) dla Lab 5.

Wczytuje dane wejściowe i dzieli je na zbiory uczący i testowy.
Obsługuje parametry ze środowiska (.env / .env.train) oraz automatycznie
wykrywa ścieżki do danych w kontenerze (/app/data) lub lokalnie (../data).
"""

import os
from pathlib import Path
from typing import Tuple, Union
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

load_dotenv()

DEFAULT_DATA_PATH = os.getenv("DATA_PATH") or "data/wine.csv"
DEFAULT_TEST_SIZE = float(os.getenv("TEST_SIZE") or "0.2")
DEFAULT_RANDOM_SEED = int(os.getenv("RANDOM_SEED") or "42")
DEFAULT_DATA_DIR = Path(os.getenv("DATA_DIR") or "data")


def _resolve_file_path(path_str: str) -> Path:
    """Wyszukuje plik w bieżącym katalogu lub w katalogu nadrzędnym."""
    p = Path(path_str)
    if p.exists():
        return p
    parent_p = Path("..") / path_str
    if parent_p.exists():
        return parent_p
    return p


def load_data(path: Union[str, Path, None] = None) -> pd.DataFrame:
    """Wczytuje zbiór danych ze wskazanej ścieżki i zwraca DataFrame."""
    if path is None or (isinstance(path, str) and path.strip() == ""):
        target_path = DEFAULT_DATA_PATH
    elif isinstance(path, Path):
        target_path = str(path)
    else:
        target_path = str(path)

    file_path = _resolve_file_path(target_path)
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
    """Dzieli DataFrame na zbiory uczący i testowy."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Oczekiwano typu pandas.DataFrame, otrzymano: {type(df)}")

    if df.empty:
        raise ValueError("Ramka danych df nie może być pusta.")

    if len(df) < 2:
        raise ValueError("Ramka danych df musi zawierać co najmniej 2 wiersze do podziału.")

    actual_test_size = DEFAULT_TEST_SIZE if test_size is None else float(test_size)
    if not (0.0 < actual_test_size < 1.0):
        raise ValueError(f"Parametr test_size musi być z przedziału (0, 1), otrzymano: {actual_test_size}")

    actual_seed = DEFAULT_RANDOM_SEED if seed is None else int(seed)

    if target_column in df.columns:
        X = df.drop(columns=[target_column])
        y = df[target_column]
    else:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

    if X.shape[1] == 0:
        raise ValueError("Ramka danych musi zawierać co najmniej jedną kolumnę cech (X).")

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
    print("=" * 60)
    print("ETAP: PRZYGOTOWANIE DANYCH (data_prep.py)")
    print("=" * 60)
    df = load_data(DEFAULT_DATA_PATH)
    print(f"Wczytano DataFrame o wymiarach: {df.shape}")
    X_train, X_test, y_train, y_test = split_data(df=df)
    print(f"Zbiór treningowy: {X_train.shape[0]}, testowy: {X_test.shape[0]}")
    print("=" * 60)


if __name__ == "__main__":
    main()
