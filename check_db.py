# check_db.py
import sqlite3

db = r"python-package\employee_events\employee_events.db"
conn = sqlite3.connect(db)
print("DB:", db)
tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()]
print("Tables:", tables)
for t in ("employee","team","employee_events","notes"):
    try:
        cnt = conn.execute(f"SELECT COUNT(*) FROM {t};").fetchone()[0]
        print(f"{t} rows: {cnt}")
    except Exception as e:
        print(f"{t} error: {e}")
conn.close()
