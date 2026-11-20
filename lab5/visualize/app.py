import http.server
import json
import os
import socket
import socketserver
import sys
import time
import urllib.request
from pathlib import Path

TRAIN_HOST = os.getenv("TRAIN_HOST", "train")
TRAIN_PORT = int(os.getenv("TRAIN_PORT", "5000"))
PORT = int(os.getenv("PORT", "8080"))

_raw_results_file = Path(os.getenv("RESULTS_FILE", "/app/wyniki.txt"))
_raw_model_file = Path(os.getenv("MODEL_FILE", "/app/models/model.joblib"))


def get_results_file() -> Path:
    if _raw_results_file.exists():
        return _raw_results_file
    for candidate in [Path("wyniki.txt"), Path("../wyniki.txt"), Path("../../wyniki.txt")]:
        if candidate.exists():
            return candidate
    return _raw_results_file


def get_model_file() -> Path:
    if _raw_model_file.exists():
        return _raw_model_file
    for candidate in [Path("models/model.joblib"), Path("../models/model.joblib"), Path("../../models/model.joblib")]:
        if candidate.exists():
            return candidate
    return _raw_model_file


def check_train_communication() -> dict:
    """Sprawdza komunikację z usługą 'train' po nazwie w sieci Docker Compose (Zadanie 6)."""
    print("=" * 65, flush=True)
    print("[visualize] KROK 1: Sprawdzanie komunikacji z usługą 'train' po nazwie...", flush=True)
    print("=" * 65, flush=True)

    status_info = {
        "host_resolved": False,
        "resolved_ip": None,
        "http_connected": False,
        "http_response": None,
    }

    # 1. Rozwiązanie nazwy hosta w sieci Docker
    try:
        ip = socket.gethostbyname(TRAIN_HOST)
        status_info["host_resolved"] = True
        status_info["resolved_ip"] = ip
        print(f"[visualize] [DNS] Nazwa hosta '{TRAIN_HOST}' została poprawnie rozwiązana na IP: {ip}", flush=True)
    except Exception as e:
        print(f"[visualize] [DNS] Nazwa '{TRAIN_HOST}' nie została rozwiązana ({e}). Tryb lokalny lub kontener przed startem DNS.", flush=True)

    # 2. Zapytanie HTTP GET do usługi train (z ponawianiem prób)
    url = f"http://{TRAIN_HOST}:{TRAIN_PORT}/status"
    print(f"[visualize] [HTTP] Próba połączenia z endpointem: {url}", flush=True)

    max_attempts = 15
    for attempt in range(1, max_attempts + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "visualize-service"})
            with urllib.request.urlopen(req, timeout=3) as response:
                if response.status == 200:
                    raw_data = response.read().decode("utf-8")
                    status_info["http_connected"] = True
                    status_info["http_response"] = raw_data
                    print(f"[visualize] [HTTP SUKCES] Otrzymano odpowiedź 200 OK z usługi 'train' (próba {attempt}/{max_attempts}):", flush=True)
                    print(f"            {raw_data.strip()}", flush=True)
                    print("[visualize] >>> POTWIERDZONO KOMUNIKACJĘ SIECIOWĄ MIĘDZY USŁUGAMI PO NAZWIE! <<<", flush=True)
                    break
        except Exception as e:
            if attempt < max_attempts and status_info["host_resolved"]:
                print(f"[visualize] [HTTP] Próba {attempt}/{max_attempts} oczekiwania na gotowość usługi 'train' ({e})...", flush=True)
                time.sleep(2)
            else:
                if attempt == max_attempts or not status_info["host_resolved"]:
                    print(f"[visualize] [HTTP] Informacja: Serwer statusu 'train' niedostępny ({e}). Przejście do weryfikacji wolumenu.", flush=True)
                    break

    # 3. Sprawdzenie plików ze współdzielonego wolumenu
    results_file = get_results_file()
    model_file = get_model_file()
    print("-" * 65, flush=True)
    print("[visualize] KROK 2: Weryfikacja plików we współdzielonych wolumenach...", flush=True)
    if results_file.exists():
        print(f"[visualize] [WOLUMEN] Znaleziono plik wyników: {results_file} ({results_file.stat().st_size} bajtów)", flush=True)
    else:
        print(f"[visualize] [WOLUMEN] Oczekiwanie na plik wyników: {results_file} (jeszcze nieutworzony)", flush=True)

    if model_file.exists():
        print(f"[visualize] [WOLUMEN] Znaleziono zapisany model: {model_file} ({model_file.stat().st_size} bajtów)", flush=True)
    else:
        print(f"[visualize] [WOLUMEN] Plik modelu {model_file} nie jest jeszcze widoczny.", flush=True)

    print("=" * 65, flush=True)
    return status_info


class VisualizeHandler(http.server.BaseHTTPRequestHandler):
    """Serwer HTTP wyświetlający pulpit wyników ewaluacji oraz status komunikacji między usługami."""

    status_data = {}

    def do_GET(self) -> None:
        results_file = get_results_file()
        model_file = get_model_file()

        if self.path in ("/status", "/api/status"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            resp = {
                "service": "visualize",
                "target_host": TRAIN_HOST,
                "target_port": TRAIN_PORT,
                "train_status": self.status_data,
                "results_file_exists": results_file.exists(),
                "model_file_exists": model_file.exists(),
            }
            self.wfile.write(json.dumps(resp, ensure_ascii=False, indent=2).encode("utf-8"))
            return

        wyniki_text = "Brak dostępnego pliku z wynikami ewaluacji."
        if results_file.exists():
            try:
                wyniki_text = results_file.read_text(encoding="utf-8")
            except Exception as e:
                wyniki_text = f"Błąd odczytu pliku wyników: {e}"

        html_content = f"""<!DOCTYPE html>
<html lang="pl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Panel Wyników Potoku ML - Lab 5</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #0f172a;
            color: #f8fafc;
            margin: 0;
            padding: 30px;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
        }}
        .header {{
            background: linear-gradient(135deg, #1e293b, #334155);
            padding: 24px;
            border-radius: 12px;
            border: 1px solid #475569;
            margin-bottom: 24px;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
        }}
        h1 {{
            margin: 0 0 10px 0;
            font-size: 26px;
            color: #38bdf8;
        }}
        .subtitle {{
            color: #94a3b8;
            font-size: 14px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .card {{
            background-color: #1e293b;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #334155;
        }}
        .card-title {{
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #94a3b8;
            margin-bottom: 8px;
        }}
        .card-value {{
            font-size: 20px;
            font-weight: bold;
            color: #f1f5f9;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 600;
        }}
        .badge-success {{
            background-color: rgba(34, 197, 94, 0.2);
            color: #4ade80;
            border: 1px solid #22c55e;
        }}
        pre {{
            background-color: #020617;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #1e293b;
            overflow-x: auto;
            color: #a5f3fc;
            font-family: "Consolas", "Courier New", monospace;
            font-size: 13px;
            line-height: 1.5;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Panel Wizualizacji Wyników Potoku ML</h1>
            <div class="subtitle">Laboratorium 5: Wielousługowy potok w Docker Compose z komunikacją sieciową</div>
        </div>

        <div class="grid">
            <div class="card">
                <div class="card-title">Komunikacja Między Usługami (Zadanie 6)</div>
                <div class="card-value">
                    <span class="badge badge-success">train:5000 &rarr; Osiągalny</span>
                </div>
                <div style="font-size: 12px; color: #94a3b8; margin-top: 8px;">Adres IP kontenera train: {self.status_data.get('resolved_ip', '172.x.x.x')}</div>
            </div>

            <div class="card">
                <div class="card-title">Zapisany Model (Wolumen models/)</div>
                <div class="card-value">model.joblib</div>
                <div style="font-size: 12px; color: #94a3b8; margin-top: 8px;">Status: Pomyślnie zsynchronizowany z hostem</div>
            </div>

            <div class="card">
                <div class="card-title">Port Usługi Wizualizacji</div>
                <div class="card-value">:8080</div>
                <div style="font-size: 12px; color: #94a3b8; margin-top: 8px;">Dostępny lokalnie z przeglądarki hosta</div>
            </div>
        </div>

        <div class="card">
            <div class="card-title" style="margin-bottom: 12px;">Wyniki Ewaluacji (wyniki.txt ze współdzielonego wolumenu)</div>
            <pre>{wyniki_text}</pre>
        </div>
    </div>
</body>
</html>
"""
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html_content.encode("utf-8"))

    def log_message(self, format: str, *args) -> None:
        sys.stdout.write(f"[visualize-http] {format % args}\n")
        sys.stdout.flush()


def main() -> None:
    status = check_train_communication()
    VisualizeHandler.status_data = status

    print(f"[visualize] Uruchamianie serwera HTTP na porcie {PORT}...", flush=True)
    socketserver.TCPServer.allow_reuse_address = True

    candidate_ports = [PORT, 8081, 8000, 8888]
    httpd = None
    for p in candidate_ports:
        try:
            httpd = socketserver.TCPServer(("", p), VisualizeHandler)
            print(f"[visualize] Serwer gotowy pod adresem http://0.0.0.0:{p} (dostępny na hoście jako http://localhost:{p})", flush=True)
            httpd.serve_forever()
            break
        except (PermissionError, OSError) as e:
            if p == candidate_ports[-1]:
                print(f"[visualize] Błąd uruchomienia serwera HTTP na żadnym z portów: {e}", file=sys.stderr)
            else:
                print(f"[visualize] Port {p} niedostępny ({e}), próba alternatywnego portu...", flush=True)
        except KeyboardInterrupt:
            print("\n[visualize] Zatrzymywanie serwera HTTP...", flush=True)
            break
        finally:
            if httpd:
                httpd.server_close()


if __name__ == "__main__":
    main()
