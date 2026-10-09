import os
import pandas as pd
from dotenv import load_dotenv
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Wczytanie zmiennych środowiskowych z pliku .env
load_dotenv()

# Odczytanie parametrów konfiguracyjnych
DATASET_NAME = os.getenv("DATASET_NAME", "iris").lower()
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
RESULTS_PATH = os.getenv("RESULTS_PATH", "wyniki.txt")

def load_data(dataset_name: str):
    """Pomocnicza funkcja do wczytywania wybranego zbioru danych."""
    if dataset_name == "wine":
        return load_wine()
    else:
        # Domyślnie iris
        return load_iris()

def main():
    print(f"Wczytane parametry z .env:")
    print(f" - DATASET_NAME: {DATASET_NAME}")
    print(f" - TEST_SIZE: {TEST_SIZE}")
    print(f" - RANDOM_SEED: {RANDOM_SEED}")
    print(f" - RESULTS_PATH: {RESULTS_PATH}\n")

    # 1. Wczytanie zbioru danych
    data = load_data(DATASET_NAME)
    X = pd.DataFrame(data=data.data, columns=data.feature_names)
    y = data.target
    target_names = data.target_names

    # 2. Podział na zbiór treningowy i testowy z użyciem zadeklarowanych parametrów
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    # 3. Trening modelu — Drzewo Decyzyjne
    dt_model = DecisionTreeClassifier(random_state=RANDOM_SEED)
    dt_model.fit(X_train, y_train)

    # 4. Predykcja na zbiorze testowym
    y_pred_dt = dt_model.predict(X_test)

    # 5. Wyliczenie metryk
    acc_dt = accuracy_score(y_test, y_pred_dt)
    report_dt = classification_report(y_test, y_pred_dt, target_names=target_names)

    # 6. Przygotowanie tekstu z wynikami
    output_text = (
        "==================================================\n"
        "EWALUACJA MODELU NA ZBIORZE TESTOWYM\n"
        "==================================================\n\n"
        f"Zbiór danych: {DATASET_NAME.upper()}\n"
        f"Parametry: TEST_SIZE={TEST_SIZE}, RANDOM_SEED={RANDOM_SEED}\n\n"
        "MODEL: DRZEWO DECYZYJNE (DecisionTreeClassifier)\n"
        "--------------------------------------------------\n"
        f"Accuracy Score: {acc_dt:.4f}\n\n"
        "Classification Report:\n"
        f"{report_dt}\n"
        "==================================================\n"
    )

    # Wyświetlenie w konsoli
    print(output_text)

    # Zapis wyników do pliku wskazanego w RESULTS_PATH
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(output_text)

    print(f"Wyniki zostały pomyślnie zapisane do pliku: {RESULTS_PATH}")

if __name__ == "__main__":
    main()
