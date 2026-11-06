"""Główny punkt wejścia i re-eksport modułu evaluate."""

import sys
from pathlib import Path

# Zapewnienie dostępu do modułów w katalogu src
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from src.evaluate import load_model, compute_metrics, format_results, save_results, main

__all__ = ["load_model", "compute_metrics", "format_results", "save_results", "main"]

if __name__ == "__main__":
    main()
