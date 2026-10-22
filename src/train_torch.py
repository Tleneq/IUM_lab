"""Moduł do trenowania, ewaluacji i walidacji krzyżowej modelu sieci neuronowej w PyTorch (Zadania 3 i 5).

Zawiera:
1. Klasę WineClassifier dziedziczącą po torch.nn.Module (2 warstwy ukryte, aktywacja ReLU).
2. Konwersję wspólnego zbioru danych z Zadania 1 do tensorów torch.Tensor (ze standaryzacją StandardScaler).
3. Wybór funkcji straty (CrossEntropyLoss) oraz optymalizatora (Adam).
4. Zadanie 3: Trening na całym zbiorze treningowym (60 epok) i ewaluację na zbiorze testowym.
5. Zadanie 5: Ręczną 5-krotną walidację krzyżową (KFold shuffle=True, n_splits=5) na zbiorze treningowym:
   - podział danych za pomocą indeksów z KFold,
   - tworzenie nowej instancji sieci i optymalizatora dla każdego foldu (aby nie trenować tych samych wag),
   - pełny cykl treningowy w każdym foldzie z osobnym dopasowaniem StandardScaler do zbioru treningowego foldu,
   - pomiar dokładności na foldzie walidacyjnym/testowym,
   - wyliczenie średniej dokładności oraz odchylenia standardowego.
"""

import os
from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dotenv import load_dotenv
from sklearn.model_selection import KFold
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
CV_FOLDS = int(os.getenv("CV_FOLDS", 5))
MODELS_DIR = Path("models")


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Ustawia ziarno losowości dla pełnej powtarzalności wyników."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class WineClassifier(nn.Module):
    """Sieć neuronowa w pełni połączona (MLP) z dwiema warstwami ukrytymi i ReLU."""

    def __init__(self, in_features: int = 13, hidden_dim1: int = 32, hidden_dim2: int = 16, num_classes: int = 3):
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


def prepare_tensors(X_train, X_test, y_train, y_test) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, StandardScaler]:
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
    verbose: bool = True,
) -> List[Dict]:
    """Pętla treningowa sieci neuronowej: forward, obliczenie straty, backward, optimizer.step()."""
    history = []
    if verbose:
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

        if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == epochs):
            print(f"Epoka {epoch:3d}/{epochs:3d} | Loss: {loss.item():.4f} | Train Acc: {train_acc:.4f}")

    if verbose:
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


def run_pytorch_cross_validation(
    X_train_df: pd.DataFrame,
    y_train_series: pd.Series,
    n_splits: int = CV_FOLDS,
    random_state: int = RANDOM_SEED,
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
):
    """Przeprowadza ręczną n-krotną walidację krzyżową (KFold) dla sieci WineClassifier."""
    X_values = X_train_df.values
    y_values = y_train_series.values
    in_features = X_values.shape[1]
    num_classes = len(np.unique(y_values))

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_accuracies = []

    print(f"\nUruchamianie ręcznej {n_splits}-krotnej walidacji krzyżowej w PyTorch...")
    print(f"Podział KFold: shuffle=True, random_state={random_state}")

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_values, y_values), start=1):
        # Ziarno dla powtarzalności inicjalizacji w ramach foldu
        set_seed(random_state + fold_idx)

        # 1. Standaryzacja dopasowana wyłącznie na zbiorze treningowym danego foldu
        scaler = StandardScaler()
        X_fold_train = scaler.fit_transform(X_values[train_idx])
        X_fold_val = scaler.transform(X_values[val_idx])
        y_fold_train = y_values[train_idx]
        y_fold_val = y_values[val_idx]

        # 2. Konwersja na tensory
        X_tr_tensor = torch.tensor(X_fold_train, dtype=torch.float32)
        y_tr_tensor = torch.tensor(y_fold_train, dtype=torch.long)
        X_va_tensor = torch.tensor(X_fold_val, dtype=torch.float32)

        # 3. NOWA INSTANCJA SIECI dla każdego foldu (aby wagi nie były przenoszone)
        fold_model = WineClassifier(in_features=in_features, hidden_dim1=32, hidden_dim2=16, num_classes=num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(fold_model.parameters(), lr=lr)

        # 4. Pełny trening na danym foldzie
        train_model(fold_model, X_tr_tensor, y_tr_tensor, criterion, optimizer, epochs=epochs, verbose=False)

        # 5. Ewaluacja na foldzie walidacyjnym/testowym
        fold_model.eval()
        with torch.no_grad():
            val_logits = fold_model(X_va_tensor)
            val_preds = torch.argmax(val_logits, dim=1).cpu().numpy()
            val_acc = float(accuracy_score(y_fold_val, val_preds))

        fold_accuracies.append(val_acc)
        print(f"  Fold {fold_idx}: Accuracy = {val_acc:.4f} ({val_acc * 100:.2f}%) [próbek walidacyjnych: {len(val_idx)}]")

    mean_acc = float(np.mean(fold_accuracies))
    std_acc = float(np.std(fold_accuracies))

    return fold_accuracies, mean_acc, std_acc


def append_or_update_results(section_header: str, section_content: str, file_path: Path):
    """Pomocnicza funkcja do aktualizowania lub dopisywania sekcji w pliku wyników."""
    existing_content = ""
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            existing_content = f.read().strip()

    if section_header in existing_content:
        parts = existing_content.split(section_header)
        prefix = parts[0].strip()
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
    print("ZADANIA 3 & 5: MODEL W PYTORCH (TRENING & RĘCZNY K-FOLD CV)")
    print("=" * 60)

    # 1. Powtarzalność losowości
    set_seed(RANDOM_SEED)

    # 2. Wczytanie wspólnego podziału danych z Zadania 1
    X_train, X_test, y_train, y_test, feature_names, target_names = get_train_test_data(data_dir=DATA_DIR)
    print(f"Wczytano dane z: {DATA_DIR}")
    print(f" - X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f" - X_test:  {X_test.shape}, y_test:  {y_test.shape}")
    print(f" - Liczba cech: {len(feature_names)}, Liczba klas: {len(target_names)}")

    # 3. Konwersja na tensory PyTorch (Zadanie 3)
    X_train_t, y_train_t, X_test_t, y_test_t, scaler = prepare_tensors(X_train, X_test, y_train, y_test)

    # 4. Inicjalizacja architektury modelu (nn.Module)
    in_features = X_train_t.shape[1]
    num_classes = len(target_names)
    model = WineClassifier(in_features=in_features, hidden_dim1=32, hidden_dim2=16, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. Trening modelu głównego (Zadanie 3)
    history = train_model(model, X_train_t, y_train_t, criterion, optimizer, epochs=EPOCHS, verbose=True)

    # 6. Ewaluacja na zbiorze testowym
    acc_torch, report_torch, matrix_torch, test_preds = evaluate_model(
        model, X_test_t, y_test.to_numpy(), target_names
    )

    output_lines_task3 = [
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
    torch_summary = "\n".join(output_lines_task3)
    print(torch_summary)

    # Zapis wag modelu do katalogu models/
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_save_path = MODELS_DIR / "torch_model.pt"
    torch.save(model.state_dict(), model_save_path)
    print(f"\nWagi modelu zapisane do: {model_save_path}")

    # --- Zadanie 5: Ręczna 5-krotna walidacja krzyżowa ---
    print("\n" + "=" * 60)
    print("ZADANIE 5: 5-KROTNA WALIDACJA KRZYŻOWA W PYTORCH (RĘCZNY K-FOLD)")
    print("=" * 60)

    cv_scores, cv_mean, cv_std = run_pytorch_cross_validation(
        X_train, y_train, n_splits=CV_FOLDS, random_state=RANDOM_SEED, epochs=EPOCHS, lr=LEARNING_RATE
    )

    fold_details = []
    for i, score in enumerate(cv_scores, start=1):
        fold_line = f"  Fold {i}: Accuracy = {score:.4f} ({score * 100:.2f}%)"
        fold_details.append(fold_line)

    print("-" * 60)
    print(f">>> ŚREDNIA DOKŁADNOŚĆ (Mean CV Accuracy PyTorch): {cv_mean:.4f} ({cv_mean * 100:.2f}%) <<<")
    print(f">>> ODCHYLENIE STANDARDOWE (Std Dev PyTorch):    {cv_std:.4f} (±{cv_std * 100:.2f}%) <<<")
    print("=" * 60)

    task5_content = (
        f"Model: WineClassifier (nn.Module - nowa instancja na każdy fold)\n"
        f"Metoda: Ręczny KFold (n_splits={CV_FOLDS}, shuffle=True, random_state={RANDOM_SEED})\n"
        f"Trening: {EPOCHS} epok, Adam(lr={LEARNING_RATE}), CrossEntropyLoss\n"
        f"Standaryzacja: StandardScaler liczony osobno wewnątrz każdego foldu\n"
        f"Zbiór: Treningowy z Zadania 1 ({len(y_train)} próbek)\n"
        f"------------------------------------------------------------\n"
        f"Wyniki dla poszczególnych foldów:\n"
        + "\n".join(fold_details)
        + f"\n------------------------------------------------------------\n"
        f">>> ŚREDNIA DOKŁADNOŚĆ (Mean CV Accuracy): {cv_mean:.4f} ({cv_mean * 100:.2f}%) <<<\n"
        f">>> ODCHYLENIE STANDARDOWE (Std Dev):    {cv_std:.4f} (±{cv_std * 100:.2f}%) <<<\n"
        f"============================================================"
    )

    header_task5 = (
        "============================================================\n"
        "WALIDACJA KRZYŻOWA PYTORCH (ZADANIE 5)"
    )

    append_or_update_results(header_task5, task5_content, RESULTS_PATH)
    print(f"\nWyniki walidacji krzyżowej PyTorch zapisane do: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
