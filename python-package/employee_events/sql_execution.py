from sqlite3 import connect
from pathlib import Path
from functools import wraps
import pandas as pd

# Using pathlib, create a `db_path` variable
# that points to the absolute path for the `employee_events.db` file
db_path = Path(__file__).resolve().parent / "employee_events.db"
db_path = str(db_path)


# OPTION 1: MIXIN
# Define a class called `QueryMixin`
class QueryMixin:
    
    # Define a method named `pandas_query`
    # that receives an sql query as a string
    # and returns the query's result
    # as a pandas dataframe
    def pandas_query(self, sql: str, params: dict = None) -> pd.DataFrame:
        """
        Execute SQL and return a pandas DataFrame.
        params: optional dict for named SQL parameters.
        """
        conn = connect(db_path)
        try:
            # pandas.read_sql_query supports sqlite3 connections and params
            df = pd.read_sql_query(sql, conn, params=params or {})
            return df
        finally:
            conn.close()

    # Define a method named `query`
    # that receives an sql_query as a string
    # and returns the query's result as
    # a list of tuples. (You will need
    # to use an sqlite3 cursor)
    def query(self, sql: str, params: tuple = None):
        """
        Execute SQL and return a list of tuples (cursor.fetchall()).
        params: optional sequence or mapping for parameter substitution.
        """
        conn = connect(db_path)
        try:
            cur = conn.cursor()
            if params is not None:
                cur.execute(sql, params)
            else:
                cur.execute(sql)
            result = cur.fetchall()
            return result
        finally:
            conn.close()
    

 
 # Leave this code unchanged
def query(func):
    """
    Decorator that runs a standard sql execution
    and returns a list of tuples
    """

    @wraps(func)
    def run_query(*args, **kwargs):
        query_string = func(*args, **kwargs)
        connection = connect(db_path)
        cursor = connection.cursor()
        result = cursor.execute(query_string).fetchall()
        connection.close()
        return result
    
    return run_query
