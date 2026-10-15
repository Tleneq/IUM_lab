import pandas as pd
from sklearn.datasets import load_iris

def main():
    # Wczytanie zbioru danych Iris z scikit-learn
    iris = load_iris()
    
    # Zamiana na obiekt pandas.DataFrame i przygotowanie do analizy
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target

    # 1. Wyświetlenie pierwszych pięciu wierszy
    print("--- Pierwsze 5 wierszy (head()) ---")
    print(df.head())
    print("\n" + "=" * 50 + "\n")

    # 2. Wyświetlenie rozmiaru zbioru danych
    print("--- Rozmiar zbioru (shape) ---")
    print(f"Wymiary (wiersze, kolumny): {df.shape}")
    print("\n" + "=" * 50 + "\n")

    # 3. Wyświetlenie podstawowych statystyk opisowych (liczność, średnia, odchylenie standardowe, min, max, kwartyle)
    print("--- Podstawowe statystyki opisowe (describe()) ---")
    print(df.describe())

if __name__ == "__main__":
    main()
