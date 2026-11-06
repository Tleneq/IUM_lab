"""Główny punkt wejścia i re-eksport modułu train."""

import sys
from pathlib import Path

# Zapewnienie dostępu do modułów w katalogu src
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from src.train import build_pipeline, train_model, save_model, main

__all__ = ["build_pipeline", "train_model", "save_model", "main"]

if __name__ == "__main__":
    main()
