# report/utils.py
from pathlib import Path
import pickle
import sqlite3
import logging
import os
from typing import Any, Optional

# Logging (sichtbar in stdout / Reviewer-Logs)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Projekt- und Asset-Pfade
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
MODEL_PATH = ASSETS_DIR / "model.pkl"

# Standard-Orte, in denen wir nach der DB suchen
DEFAULT_DB_LOCATIONS = [
    PROJECT_ROOT / "employee_events.db",        # Projekt-Root (häufig während Entwicklung)
    ASSETS_DIR / "employee_events.db",         # assets/ (falls dort abgelegt)
    PROJECT_ROOT / "data" / "employee_events.db",  # alternative Struktur
]

def load_model(path: Optional[Path] = None) -> Any:
    """
    Lädt model.pkl aus assets (wie vorher). Loggt Pfad und wirft klare Fehlermeldung.
    """
    p = Path(path) if path else MODEL_PATH
    logger.info("Trying to load model from %s", p)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found at {p}")
    with open(p, "rb") as f:
        return pickle.load(f)

def get_db_path(explicit: Optional[Path] = None) -> Path:
    """
    Liefert den bestmöglichen Pfad zur employee_events.db.
    Reihenfolge:
     1) explizit übergeben
     2) ENV VAR EMP_EVENTS_DB
     3) DEFAULT_DB_LOCATIONS (Projekt root, assets, data/…)
    Wenn nichts gefunden wird, wird FileNotFoundError mit hilfreicher Meldung geworfen.
    """
    # 1) explizit
    if explicit:
        p = Path(explicit)
        logger.info("Using explicit DB path: %s", p)
        if p.exists():
            return p
        raise FileNotFoundError(f"Explicit DB path does not exist: {p}")

    # 2) env var
    env = os.environ.get("EMP_EVENTS_DB")
    if env:
        p = Path(env)
        logger.info("Trying DB path from EMP_EVENTS_DB env var: %s", p)
        if p.exists():
            return p
        logger.warning("EMP_EVENTS_DB set but file not found: %s", p)

    # 3) default locations
    for candidate in DEFAULT_DB_LOCATIONS:
        logger.info("Checking DB candidate: %s", candidate)
        if candidate.exists():
            logger.info("Found DB at %s", candidate)
            return candidate

    # 4) not found -> hilfreiche Fehlermeldung
    search_list = "\n".join(str(x) for x in DEFAULT_DB_LOCATIONS)
    raise FileNotFoundError(
        "employee_events.db konnte nicht gefunden werden. Searched locations:\n"
        f"{search_list}\n\n"
        "Lösungen:\n"
        "- Lege employee_events.db in dein Projekt-Root oder assets/ ab\n"
        "- Oder setze ENV var EMP_EVENTS_DB auf den absoluten Pfad\n"
        "- Oder übergebe den Pfad explizit an connect_db(path)\n"
    )

def connect_db(path: Optional[Path] = None, **sqlite_kwargs) -> sqlite3.Connection:
    """
    Liefert eine sqlite3.Connection; nutzt check_same_thread=False standardmäßig
    (nützlich für Dash/Flask multithreading).
    Benutze: conn = connect_db() oder conn = connect_db("/abs/path/employee_events.db")
    """
    p = get_db_path(path)
    kwargs = dict(check_same_thread=False)
    kwargs.update(sqlite_kwargs)
    logger.info("Opening sqlite DB at %s (sqlite kwargs: %s)", p, kwargs)
    conn = sqlite3.connect(p, **kwargs)
    return conn
