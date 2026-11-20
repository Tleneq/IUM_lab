# ==============================================================================
# Dockerfile - Potok treningowy ML (Lab 5, Zadanie 1)
# ==============================================================================

# Obraz bazowy z lekką dystrybucją Pythona
FROM python:3.11-slim

# Katalog roboczy wewnątrz kontenera
WORKDIR /app

# Optymalizacja cache warstw: instalacja zależności przed kopiowaniem kodu źródłowego
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Skopiowanie modułów potoku uczenia maszynowego oraz pliku wzorcowego konfiguracji
COPY data_prep.py train.py evaluate.py .env.example ./

# Utworzenie domyślnych katalogów na dane i modele
RUN mkdir -p /app/data /app/models

# Domyślne polecenie uruchamiające proces trenowania modelu
CMD ["python", "train.py"]
