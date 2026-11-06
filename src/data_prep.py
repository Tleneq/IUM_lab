"""Moduł do wczytywania i przygotowania danych (data_prep.py).

Zadanie 1:
Separacja odpowiedzialności - moduł odpowiedzialny za wczytywanie i podział danych.
Funkcje load_data i split_data są czyste (brak efektów ubocznych, brak print/zapisu).
Dopiero wywołanie pliku jako skryptu (CLI) wczytuje zmienne z .env i zapisuje wyniki podziału.
"""

import os
from pathlib import Path
from typing import Tuple, Union
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split


def load_data(path: Union[str, Path, None] = None) -> pd.DataFrame:
    """Wczytuje zbiór danych ze wskazanej ścieżki i zwraca DataFrame.

    Funkcja nie posiada efektów ubocznych (brak printów i zapisu do pliku).

    Args:
        path: Ścieżka do pliku z danymi (np. CSV). Jeśli None, pobiera DATA_PATH z .env lub domyślną ścieżkę.

    Returns:
        pd.DataFrame: Wczytane dane.

    Raises:
        FileNotFoundError: Gdy wskazany plik nie istnieje.
        TypeError: Gdy wczytany obiekt nie jest ramką DataFrame.
    """
    if path is None:
        load_dotenv()
        env_path = os.getenv("DATA_PATH")
        file_path = Path(env_path) if env_path else Path("data/wine.csv")
    elif isinstance(path, Path):
        file_path = path
    else:
        file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Plik danych nie został znaleziony: {file_path}")

    df = pd.read_csv(file_path)
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Oczekiwano typu pandas.DataFrame, otrzymano: {type(df)}")

    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
    target_column: str = "target",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Dzieli DataFrame na zbiory uczący i testowy.

    Funkcja nie posiada efektów ubocznych (brak printów i zapisu do pliku).

    Args:
        df: Ramka danych zawierająca cechy i kolumnę docelową.
        test_size: Udział zbioru testowego (domyślnie 0.2).
        seed: Ziarno losowości (domyślnie 42).
        target_column: Nazwa kolumny etykiety/celu (domyślnie 'target').
                       Jeśli kolumny nie ma w DataFrame, za cel przyjmuje się ostatnią kolumnę.

    Returns:
        Krotka (X_train, X_test, y_train, y_test).
    """
    if target_column in df.columns:
        X = df.drop(columns=[target_column])
        y = df[target_column]
    else:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

    # Bezpieczna stratyfikacja (jeśli każda klasa ma co najmniej 2 instancje)
    stratify = None
    if hasattr(y, "value_counts"):
        counts = y.value_counts()
        if len(counts) > 1 and (counts >= 2).all():
            stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )

    return X_train, X_test, y_train, y_test


def main() -> None:
    """Główna funkcja CLI do przygotowania danych w potoku CI/CD.

    Wczytuje parametry z pliku .env, przetwarza dane i zapisuje zbiory do katalogu data/.
    """
    load_dotenv()

    data_path = os.getenv("DATA_PATH", "data/wine.csv")
    test_size = float(os.getenv("TEST_SIZE", "0.2"))
    random_seed = int(os.getenv("RANDOM_SEED", "42"))
    data_dir = Path(os.getenv("DATA_DIR", "data"))

    print("=" * 60)
    print("ETAP: PRZYGOTOWANIE DANYCH (data_prep.py)")
    print("=" * 60)
    print(f"Ścieżka danych wejściowych: {data_path}")
    print(f"Parametry: test_size={test_size}, random_seed={random_seed}")
    print("-" * 60)

    # 1. Wczytanie danych
    df = load_data(data_path)
    print(f"Wczytano DataFrame o wymiarach: {df.shape}")

    # 2. Podział danych
    X_train, X_test, y_train, y_test = split_data(
        df=df,
        test_size=test_size,
        seed=random_seed,
    )

    # 3. Zapis plików dla kolejnych etapów potoku (train.py / evaluate.py)
    data_dir.mkdir(parents=True, exist_ok=True)

    train_df = X_train.copy()
    train_df["target"] = y_train
    train_df.to_csv(data_dir / "train.csv", index=False)

    test_df = X_test.copy()
    test_df["target"] = y_test
    test_df.to_csv(data_dir / "test.csv", index=False)

    X_train.to_csv(data_dir / "X_train.csv", index=False)
    X_test.to_csv(data_dir / "X_test.csv", index=False)
    y_train.to_csv(data_dir / "y_train.csv", index=False)
    y_test.to_csv(data_dir / "y_test.csv", index=False)

    total = len(df)
    print(f"Zbiór treningowy: {X_train.shape[0]} próbek ({X_train.shape[0] / total:.1%})")
    print(f"Zbiór testowy:    {X_test.shape[0]} próbek ({X_test.shape[0] / total:.1%})")
    print(f"Zapisano pliki do katalogu: {data_dir.resolve()}")
    print(" - train.csv, test.csv")
    print(" - X_train.csv, y_train.csv, X_test.csv, y_test.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
