# python-package/employee_events/employee.py
from .query_base import QueryBase
from .sql_execution import QueryMixin  # falls QueryBase es nicht schon hat

# try to obtain connect_db helper from report.utils (if present)
try:
    from report.utils import connect_db
except Exception:
    connect_db = None


class Employee(QueryBase):
    """
    Employee model adapted to the package DB schema (tables: employee, team, employee_events, notes).
    """

    name = "employee"  # keep this if other code relies on it

    def __init__(self, *args, **kwargs):
        # call parent init if present
        try:
            super().__init__(*args, **kwargs)
        except Exception:
            pass

        # try to attach a package-provided DB connection (non-fatal)
        try:
            if connect_db is not None:
                self.conn = connect_db()
        except Exception:
            pass

    def names(self):
        """
        Robust: versuche zuerst first_name || ' ' || last_name,
        und falls das wegen fehlender Spalten in der DB fehlschlägt,
        versuche als Fallback full_name.
        """
        sql_preferred = """
        SELECT first_name || ' ' || last_name AS full_name, employee_id
        FROM employee
        ORDER BY employee_id;
        """
        sql_fallback = """
        SELECT full_name, employee_id
        FROM employee
        ORDER BY employee_id;
        """
        try:
            return self.query(sql_preferred)
        except Exception:
            # fallback auf full_name (falls vorhanden)
            return self.query(sql_fallback)

    def username(self, id):
        """
        Robust: wie names(), aber für einen einzelnen employee_id.
        """
        try:
            eid = int(id)
        except Exception:
            eid = -1

        sql_preferred = f"""
        SELECT first_name || ' ' || last_name AS full_name
        FROM employee
        WHERE employee_id = {eid};
        """
        sql_fallback = f"""
        SELECT full_name
        FROM employee
        WHERE employee_id = {eid};
        """
        try:
            return self.query(sql_preferred)
        except Exception:
            return self.query(sql_fallback)

    def model_data(self, id):
        """
        Return a pandas.DataFrame with one row and two columns:
        - 'positive'
        - 'negative'
        These are numeric counts suitable as ML features.
        This function tries multiple DB schemas and always returns a DataFrame.
        """
        import pandas as pd
        try:
            eid = int(id)
        except Exception:
            eid = -1

        # 1) preferred: employee_events has aggregated columns positive_events / negative_events
        try:
            sql = f"""
            SELECT
                COALESCE(SUM(positive_events), 0) AS positive,
                COALESCE(SUM(negative_events), 0) AS negative
            FROM employee_events
            WHERE employee_id = {eid};
            """
            df = self.pandas_query(sql)
            if df is not None:
                # Ensure column names are exactly 'positive' and 'negative'
                df = df.rename(columns={c: c.lower() for c in df.columns})
                # if pandas_query returns columns like 'positive'/'negative' already, fine
                if "positive" in df.columns and "negative" in df.columns:
                    return df.fillna(0)
        except Exception:
            pass

        # 2) fallback: maybe employee table contains aggregated columns
        try:
            sql2 = f"""
            SELECT
                COALESCE(positive_events, 0) AS positive,
                COALESCE(negative_events, 0) AS negative
            FROM employee
            WHERE employee_id = {eid}
            LIMIT 1;
            """
            df2 = self.pandas_query(sql2)
            if df2 is not None:
                df2 = df2.rename(columns={c: c.lower() for c in df2.columns})
                if "positive" in df2.columns and "negative" in df2.columns:
                    return df2.fillna(0)
        except Exception:
            pass

        # 3) last resort: sum numeric candidate columns from raw employee_events rows
        try:
            tbl = self.pandas_query(f"SELECT * FROM employee_events WHERE employee_id = {eid};")
            if tbl is not None and not getattr(tbl, "empty", True):
                cols = [c.lower() for c in tbl.columns]
                pos_col = next((c for c in tbl.columns if c.lower() in ("positive_events","positive_count","positive","pos")), None)
                neg_col = next((c for c in tbl.columns if c.lower() in ("negative_events","negative_count","negative","neg")), None)
                pos_sum = float(tbl[pos_col].sum()) if pos_col else 0.0
                neg_sum = float(tbl[neg_col].sum()) if neg_col else 0.0
                return pd.DataFrame([{"positive": pos_sum, "negative": neg_sum}])
        except Exception:
            pass

        # 4) safe fallback: zeros so predictor can still run (will produce low-risk prediction)
        return pd.DataFrame([{"positive": 0.0, "negative": 0.0}])
