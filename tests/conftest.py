# tests/conftest.py
import sys
from pathlib import Path

# Caminho da raiz do projeto: .../ES2-Atv5_ml-metrics-python
ROOT_DIR = Path(__file__).resolve().parents[1]

# Garante que a raiz do projeto está no sys.path
sys.path.insert(0, str(ROOT_DIR))
