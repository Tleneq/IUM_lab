FROM python:3.11-slim

WORKDIR /app

# Kopiowanie statycznego pliku HTML do katalogu serwera
COPY index.html .

# Dokumentacja portu, na którym nasłuchuje serwer HTTP
EXPOSE 8000

# Uruchomienie wbudowanego serwera HTTP Pythona
CMD ["python", "-m", "http.server", "8000"]
