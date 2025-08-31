# Import any dependencies needed to execute sql queries
from .sql_execution import QueryMixin
import pandas as pd

# Define a class called QueryBase
# Use inheritance to add methods
# for querying the employee_events database.
class QueryBase(QueryMixin):
    """Base class for entity-specific query classes."""

    # Create a class attribute called `name`
    # set the attribute to an empty string
    name = ""

    # Define a `names` method that receives
    # no passed arguments
    def names(self):
        # Default implementation: empty list
        return []

    # Define an `event_counts` method
    # that receives an `id` argument
    # This method should return a pandas dataframe
    def event_counts(self, id):
        """
        QUERY 1
        Return daily aggregated positive/negative event counts for the entity
        identified by {id}. The query uses the class attribute `name` to
        build the correct column/table names (e.g., employee_id or team_id).
        """
        sql = f"""
        SELECT
            date(event_date) AS day,
            SUM(positive_events) AS positive_events,
            SUM(negative_events) AS negative_events
        FROM employee_events
        WHERE {self.name}_id = {id}
        GROUP BY day
        ORDER BY day;
        """
        return self.pandas_query(sql)

    # Define a `notes` method that receives an id argument
    # This function should return a pandas dataframe
    def notes(self, id):
        """
        QUERY 2
        Return note_date and note from notes table for the given entity id.
        The notes table stores notes for both employees and teams, so the
        id column name depends on self.name (employee_id or team_id).
        """
        sql = f"""
        SELECT
            note_date,
            note
        FROM notes
        WHERE {self.name}_id = {id}
        ORDER BY note_date;
        """
        return self.pandas_query(sql)
