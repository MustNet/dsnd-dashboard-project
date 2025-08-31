# tests/test_employee_events.py
from pathlib import Path
import sqlite3
import pytest

DB_PATH = Path(__file__).resolve().parent.parent / "python-package" / "employee_events" / "employee_events.db"

@pytest.fixture
def db_path():
    return DB_PATH

def test_db_exists(db_path):
    assert db_path.exists(), f"DB file not found at {db_path}"

def _get_table_names(db_path):
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        rows = cur.fetchall()
        return {r[0] for r in rows}
    finally:
        conn.close()

def test_employee_table_exists(db_path):
    tables = _get_table_names(db_path)
    assert "employee" in tables or "employees" in tables

def test_team_table_exists(db_path):
    tables = _get_table_names(db_path)
    assert "team" in tables

def test_employee_events_table_exists(db_path):
    tables = _get_table_names(db_path)
    assert "employee_events" in tables
