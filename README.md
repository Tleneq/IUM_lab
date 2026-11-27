# IUM_lab - Machine Learning Pipeline

[![CI](https://github.com/Tleneq/IUM_lab/actions/workflows/ci.yml/badge.svg)](https://github.com/Tleneq/IUM_lab/actions/workflows/ci.yml)

Projekt potoku uczenia maszynowego (ML Pipeline) realizowany w ramach przedmiotu Inżynieria Uczenia Maszynowego.

## Struktura projektu
- `src/` – moduły potoku ML (`data_prep.py`, `train.py`, `evaluate.py`)
- `tests/` – testy jednostkowe weryfikowane przez `pytest`
- `data/` – zbiory danych wejściowych (`wine.csv`)
- `models/` – wytrenowane artefakty modeli scikit-learn (`model.joblib`)
- `.github/workflows/ci.yml` – pipeline ciągłej integracji GitHub Actions

## Uruchomienie lokalne
```bash
# Uruchomienie testów jednostkowych
pytest

# Trenowanie potoku ML
python src/train.py
```
