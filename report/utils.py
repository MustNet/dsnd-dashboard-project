# report/utils.py
from pathlib import Path
import pickle
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "assets" / "model.pkl"

def load_model(path: Optional[Path] = None) -> Any:
    p = path or MODEL_PATH
    if not p.exists():
        raise FileNotFoundError(f"Model file not found at {p}")
    with open(p, "rb") as f:
        return pickle.load(f)
