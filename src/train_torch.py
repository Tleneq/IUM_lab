"""Moduł do trenowania i ewaluacji modelu sieci neuronowej w PyTorch (Zadanie 3).

Zawiera:
1. Klasę WineClassifier dziedziczącą po torch.nn.Module (2 warstwy ukryte, aktywacja ReLU).
2. Konwersję wspólnego zbioru danych z Zadania 1 do tensorów torch.Tensor (z uwzględnieniem StandardScaler).
3. Wybór funkcji straty (CrossEntropyLoss) oraz optymalizatora (Adam).
4. Pętlę treningową obejmującą kilkadziesiąt epok (forward, strata, backward, optimizer.step()).
5. Ewaluację dokładności (accuracy_score) na zbiorze testowym w porównaniu do Zadania 2.
"""

import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from load_dataset import get_train_test_data

# Załadowanie zmiennych konfiguracyjnych
load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))
RESULTS_PATH = Path(os.getenv("RESULTS_PATH", "wyniki.txt"))
EPOCHS = int(os.getenv("EPOCHS", 60))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 0.01))
MODELS_DIR = Path("models")


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Ustawia ziarno losowości dla pełnej powtarzalności wyników."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class WineClassifier(nn.Module):
    """Sieć neuronowa w pełni połączona (MLP) z dwiema warstwami ukrytymi i ReLU."""

    def __init__(self, in_features: int, hidden_dim1: int = 32, hidden_dim2: int = 16, num_classes: int = 3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Przejście w przód (forward pass) zwracające logity klas."""
        return self.network(x)


def prepare_tensors(X_train, X_test, y_train, y_test):
    """Skaluje cechy za pomocą StandardScaler i konwertuje dane do tensorów PyTorch."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.long)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.long)

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, scaler


def train_model(
    model: nn.Module,
    X_train_tensor: torch.Tensor,
    y_train_tensor: torch.Tensor,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int = EPOCHS,
) -> list:
    """Pętla treningowa sieci neuronowej: forward, obliczenie straty, backward, optimizer.step()."""
    history = []
    print(f"\n--- Rozpoczęcie treningu sieci ({epochs} epok) ---")

    for epoch in range(1, epochs + 1):
        model.train()

        # 1. Zerowanie gradientów
        optimizer.zero_grad()

        # 2. Forward pass
        outputs = model(X_train_tensor)

        # 3. Obliczenie straty
        loss = criterion(outputs, y_train_tensor)

        # 4. Backward pass (propagacja wsteczna)
        loss.backward()

        # 5. Krok optymalizatora (aktualizacja wag)
        optimizer.step()

        # Obliczenie dokładności treningowej
        with torch.no_grad():
            preds = torch.argmax(outputs, dim=1)
            train_acc = (preds == y_train_tensor).float().mean().item()

        history.append({"epoch": epoch, "loss": loss.item(), "train_acc": train_acc})

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"Epoka {epoch:3d}/{epochs:3d} | Loss: {loss.item():.4f} | Train Acc: {train_acc:.4f}")

    print("--- Zakończono trening ---\n")
    return history


def evaluate_model(model: nn.Module, X_test_tensor: torch.Tensor, y_test: np.ndarray, target_names: list):
    """Ewaluacja modelu na zbiorze testowym za pomocą predict (argmax z logitów)."""
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_tensor)
        test_preds = torch.argmax(test_logits, dim=1).cpu().numpy()

    acc = accuracy_score(y_test, test_preds)
    report = classification_report(y_test, test_preds, target_names=target_names)
    matrix = confusion_matrix(y_test, test_preds)

    return acc, report, matrix, test_preds


def main():
    print("=" * 60)
    print("ZADANIE 3: MODEL W PYTORCH (MULTI-LAYER PERCEPTRON)")
    print("=" * 60)

    # 1. Powtarzalność losowości
    set_seed(RANDOM_SEED)

    # 2. Wczytanie wspólnego podziału danych z Zadania 1
    X_train, X_test, y_train, y_test, feature_names, target_names = get_train_test_data(data_dir=DATA_DIR)
    print(f"Wczytano dane z: {DATA_DIR}")
    print(f" - X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f" - X_test:  {X_test.shape}, y_test:  {y_test.shape}")
    print(f" - Liczba cech: {len(feature_names)}, Liczba klas: {len(target_names)}")

    # 3. Konwersja na tensory PyTorch
    X_train_t, y_train_t, X_test_t, y_test_t, scaler = prepare_tensors(X_train, X_test, y_train, y_test)
    print(f"Dane przekształcone do tensorów torch.Tensor (typ float32 / int64)")

    # 4. Inicjalizacja architektury modelu (nn.Module)
    in_features = X_train_t.shape[1]
    num_classes = len(target_names)
    model = WineClassifier(in_features=in_features, hidden_dim1=32, hidden_dim2=16, num_classes=num_classes)
    print(f"\nArchitektura modelu:\n{model}")

    # 5. Wybór funkcji straty i optymalizatora
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print(f"Funkcja straty: {criterion.__class__.__name__}")
    print(f"Optymalizator: {optimizer.__class__.__name__} (lr={LEARNING_RATE})")

    # 6. Pętla treningowa
    history = train_model(model, X_train_t, y_train_t, criterion, optimizer, epochs=EPOCHS)

    # 7. Ewaluacja na zbiorze testowym
    acc_torch, report_torch, matrix_torch, test_preds = evaluate_model(
        model, X_test_t, y_test.to_numpy(), target_names
    )

    output_lines = [
        "=" * 60,
        "EWALUACJA MODELU PYTORCH (ZADANIE 3)",
        "=" * 60,
        f"Model: WineClassifier (nn.Module: Linear(13->32) -> ReLU -> Linear(32->16) -> ReLU -> Linear(16->3))",
        f"Funkcja straty: CrossEntropyLoss | Optymalizator: Adam(lr={LEARNING_RATE})",
        f"Liczba epok: {EPOCHS} | RANDOM_SEED={RANDOM_SEED}",
        f"Rozmiar zbioru testowego: {len(y_test)} próbek",
        "-" * 60,
        f">>> ACCURACY SCORE (PyTorch): {acc_torch:.4f} ({acc_torch * 100:.2f}%) <<<",
        "-" * 60,
        "Classification Report:",
        report_torch,
        "Confusion Matrix:",
        str(matrix_torch),
        "=" * 60,
    ]
    torch_summary = "\n".join(output_lines)
    print(torch_summary)

    # 8. Zapis wag modelu do katalogu models/
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_save_path = MODELS_DIR / "torch_model.pt"
    torch.save(model.state_dict(), model_save_path)
    print(f"\nWagi modelu zapisane do: {model_save_path}")

    # 9. Aktualizacja pliku wyniki.txt z podsumowaniem obu modeli
    existing_content = ""
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            existing_content = f.read().strip()

    # Jeśli w pliku jest już ewaluacja scikit-learn, dołączamy wyniki PyTorch
    if "EWALUACJA MODELU SCIKIT-LEARN" in existing_content:
        # Usunięcie ewentualnych poprzednich wpisów PyTorch
        if "EWALUACJA MODELU PYTORCH" in existing_content:
            parts = existing_content.split("============================================================\nEWALUACJA MODELU PYTORCH")
            existing_content = parts[0].strip()

        full_results = (
            f"{existing_content}\n\n"
            f"{torch_summary}\n\n"
            f"============================================================\n"
            f"PORÓWNANIE MODELI (ZADANIE 2 vs ZADANIE 3)\n"
            f"============================================================\n"
            f"Zbiór danych: WINE (13 cech, 3 klasy)\n"
            f"- Model 1 (scikit-learn - LogisticRegression):  0.9722 (97.22%)\n"
            f"- Model 2 (PyTorch - WineClassifier MLP):        {acc_torch:.4f} ({acc_torch * 100:.2f}%)\n"
            f"============================================================\n"
        )
    else:
        full_results = torch_summary + "\n"

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(full_results)

    print(f"Zaktualizowano plik z wynikami: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
