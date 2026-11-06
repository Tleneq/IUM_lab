"""Główny punkt wejścia i re-eksport modułu data_prep."""

import sys
from pathlib import Path

# Zapewnienie dostępu do modułów w katalogu src
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from src.data_prep import load_data, split_data, main

__all__ = ["load_data", "split_data", "main"]

if __name__ == "__main__":
    main()
