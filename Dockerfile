FROM python:3.11-slim

WORKDIR /app

# Domyślna wartość zmiennej środowiskowej GREETING (Zadanie 2)
ENV GREETING="Witaj z kontenera Docker!"

# Kopiowanie skryptu aplikacji do katalogu roboczego
COPY app.py .

# Domyślne polecenie uruchamiające aplikację
CMD ["python", "app.py"]
