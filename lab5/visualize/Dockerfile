# ==============================================================================
# Dockerfile - Usługa wizualizacji wyników i komunikacji sieciowej (Zadanie 4 i 6)
# ==============================================================================
FROM python:3.11-slim

WORKDIR /app

# Kopiowanie skryptu serwera wizualizacji
COPY app.py .

# Dokumentacja portu usługi
EXPOSE 8080

# Uruchomienie skryptu sprawdzającego komunikację i serwującego pulpit HTTP
CMD ["python", "app.py"]
