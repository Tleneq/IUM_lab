import os
import sys
import time
from pathlib import Path

def main():
    # Odczyt zmiennej środowiskowej GREETING z domyślną wartością
    greeting = os.getenv("GREETING", "Witaj z aplikacji kontenerowej!")
    print(greeting, flush=True)

    # Ścieżka do katalogu zapisu (domyślnie /data w kontenerze)
    data_dir_path = os.getenv("DATA_DIR", "/data")
    data_dir = Path(data_dir_path)

    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        wynik_file = data_dir / "wynik.txt"
        wynik_file.write_text(f"{greeting}\n", encoding="utf-8")
    except Exception as e:
        # Fallback dla uruchomień lokalnych poza kontenerem w przypadku braku uprawnień do /data
        fallback_dir = Path("./data")
        fallback_dir.mkdir(parents=True, exist_ok=True)
        fallback_file = fallback_dir / "wynik.txt"
        fallback_file.write_text(f"{greeting}\n", encoding="utf-8")
        print(f"[Ostrzeżenie] Nie można zapisać do {data_dir_path} ({e}). Zapisano w {fallback_file}", file=sys.stderr, flush=True)

    # Opcjonalne uśpienie dla testowania kontenera w tle (docker run -d / docker ps / docker stop)
    sleep_seconds = int(os.getenv("SLEEP_SECONDS", "0"))
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)

if __name__ == "__main__":
    main()
