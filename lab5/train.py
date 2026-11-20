"""Moduł do trenowania modelu scikit-learn (train.py) dla Lab 5.

Zadania 1-4, 6:
Wczytuje parametry z pliku .env / .env.train.
Dzieli dane, trenuje pipeline StandardScaler -> LogisticRegression / RandomForest,
zapisuje model do katalogu models/model.joblib, uruchamia ewaluację oraz
udostępnia punkt kontrolny HTTP dla komunikacji między usługami.
"""

import http.server
import json
import os
import socketserver
import sys
import threading
import time
from pathlib import Path
from typing import Any, Union
import joblib
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
for path_entry in (current_dir, parent_dir):
    if str(path_entry) not in sys.path:
        sys.path.insert(0, str(path_entry))

from data_prep import load_data, split_data

load_dotenv()

DEFAULT_DATA_PATH = os.getenv("DATA_PATH") or "data/wine.csv"
DEFAULT_TEST_SIZE = float(os.getenv("TEST_SIZE") or "0.2")
DEFAULT_RANDOM_SEED = int(os.getenv("RANDOM_SEED") or "42")
DEFAULT_MODEL_TYPE = os.getenv("MODEL_TYPE") or "logistic_regression"
DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH") or "models/model.joblib"
DEFAULT_MAX_ITER = int(os.getenv("MAX_ITER") or "1000")
DEFAULT_N_ESTIMATORS = int(os.getenv("N_ESTIMATORS") or "100")


def build_pipeline(
    model_type: Union[str, None] = None,
    random_state: Union[int, None] = None,
    max_iter: Union[int, None] = None,
    n_estimators: Union[int, None] = None,
) -> Pipeline:
    actual_model_type = (DEFAULT_MODEL_TYPE if model_type is None else model_type).lower()
    actual_random_state = DEFAULT_RANDOM_SEED if random_state is None else int(random_state)
    actual_max_iter = DEFAULT_MAX_ITER if max_iter is None else int(max_iter)
    actual_n_estimators = DEFAULT_N_ESTIMATORS if n_estimators is None else int(n_estimators)

    if actual_model_type in {"rf", "random_forest", "randomforestclassifier"}:
        classifier = RandomForestClassifier(
            n_estimators=actual_n_estimators,
            random_state=actual_random_state,
        )
    else:
        classifier = LogisticRegression(
            max_iter=actual_max_iter,
            random_state=actual_random_state,
        )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", classifier),
    ])
    return pipeline


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: Union[str, None] = None,
    random_state: Union[int, None] = None,
    max_iter: Union[int, None] = None,
    n_estimators: Union[int, None] = None,
) -> Pipeline:
    pipeline = build_pipeline(
        model_type=model_type,
        random_state=random_state,
        max_iter=max_iter,
        n_estimators=n_estimators,
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def save_model(model: Any, model_path: Union[str, Path, None] = None) -> Path:
    target = DEFAULT_MODEL_PATH if model_path is None else model_path
    file_path = Path(target)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, file_path)

    parent_path = Path("..") / target
    if Path("..").resolve() != file_path.resolve().parent and Path("../.git").exists():
        try:
            parent_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, parent_path)
        except Exception:
            pass

    return file_path


class _TrainingStatusHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path in ("/status", "/", "/health"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            payload = {
                "status": "ready",
                "service": "train",
                "model_path": str(DEFAULT_MODEL_PATH),
                "model_type": str(DEFAULT_MODEL_TYPE),
                "message": "Trening modelu scikit-learn zakonczony pomyslnie.",
            }
            self.wfile.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        sys.stdout.write(f"[train-http] {self.address_string()} - {format % args}\n")
        sys.stdout.flush()


def run_status_server(port: int = 5000) -> socketserver.TCPServer:
    socketserver.TCPServer.allow_reuse_address = True
    server = socketserver.TCPServer(("", port), _TrainingStatusHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    return server


def main() -> None:
    print("=" * 60)
    print("ETAP: TRENOWANIE MODELU (train.py)")
    print("=" * 60)
    print(f"Dane wejściowe: {DEFAULT_DATA_PATH}")
    print(f"Typ modelu: {DEFAULT_MODEL_TYPE} (seed={DEFAULT_RANDOM_SEED})")
    print(f"Hiperparametry: max_iter={DEFAULT_MAX_ITER}, n_estimators={DEFAULT_N_ESTIMATORS}")
    print(f"Docelowa ścieżka modelu: {DEFAULT_MODEL_PATH}")
    print("-" * 60)

    df = load_data(DEFAULT_DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(
        df=df,
        test_size=DEFAULT_TEST_SIZE,
        seed=DEFAULT_RANDOM_SEED,
    )
    print(f"Rozmiar zbioru treningowego: {X_train.shape[0]} próbek, {X_train.shape[1]} cech")

    pipeline = train_model(
        X_train=X_train,
        y_train=y_train,
        model_type=DEFAULT_MODEL_TYPE,
        random_state=DEFAULT_RANDOM_SEED,
        max_iter=DEFAULT_MAX_ITER,
        n_estimators=DEFAULT_N_ESTIMATORS,
    )
    clf_name = pipeline.named_steps["classifier"].__class__.__name__
    print(f"Wytrenowano Pipeline: StandardScaler -> {clf_name}")

    saved_path = save_model(pipeline, DEFAULT_MODEL_PATH)
    print(f"Model pomyślnie zapisany do: {saved_path.resolve()}")
    print("=" * 60)

    run_evaluate = os.getenv("RUN_EVALUATE", "true").lower() in ("true", "1", "yes")
    if run_evaluate:
        try:
            from evaluate import main as evaluate_main
            print("\n" + "=" * 60)
            print("ETAP: EWALUACJA MODELU W POTOKU (evaluate.py)")
            print("=" * 60)
            evaluate_main()
        except Exception as e:
            print(f"[Ostrzeżenie] Nie udało się wykonać automatycznej ewaluacji: {e}", file=sys.stderr)

    keep_alive = int(os.getenv("KEEP_ALIVE_SECONDS", "0"))
    status_port = int(os.getenv("STATUS_SERVER_PORT", "5000"))

    if keep_alive > 0 or os.getenv("ENABLE_STATUS_SERVER", "false").lower() in ("true", "1"):
        server = run_status_server(port=status_port)
        print(f"\n[train] Serwer statusu HTTP nasłuchuje na porcie {status_port} (dostępny jako http://train:{status_port}/status).")
        if keep_alive > 0:
            print(f"[train] Utrzymanie kontenera przez {keep_alive}s na potrzeby 'docker stats' i zapytań z usługi visualize...")
            try:
                time.sleep(keep_alive)
            except KeyboardInterrupt:
                pass
            finally:
                server.shutdown()
                print("[train] Czas oczekiwania upłynął. Zatrzymywanie usługi train.")


if __name__ == "__main__":
    main()
