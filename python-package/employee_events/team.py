# Import the QueryBase class
from .query_base import QueryBase

# Import dependencies for sql execution
from .sql_execution import QueryMixin  # optional, QueryBase already inherits it

# Create a subclass of QueryBase called `Team`
class Team(QueryBase):
    # Set the class attribute `name` to the string "team"
    name = "team"

    # Define a `names` method that receives no arguments
    # This method should return a list of tuples from an sql execution
    def names(self):
        """
        Query 5
        Return team_name and team_id for all teams.
        """
        sql = """
        SELECT team_name, team_id
        FROM team;
        """
        return self.query(sql)

    # Define a `username` method that receives an ID argument
    # This method should return a list of tuples from an sql execution
    def username(self, id):
        """
        Query 6
        Return the team_name for the given team id.
        """
        sql = f"""
        SELECT team_name
        FROM team
        WHERE team_id = {id};
        """
        return self.query(sql)

    # model_data should execute its SQL and return a pandas dataframe
    def model_data(self, id):
        sql = f"""
            SELECT positive_events, negative_events FROM (
                    SELECT employee_id
                         , SUM(positive_events) positive_events
                         , SUM(negative_events) negative_events
                    FROM {self.name}
                    JOIN employee_events
                        USING({self.name}_id)
                    WHERE {self.name}.{self.name}_id = {id}
                    GROUP BY employee_id
                   )
                """
        return self.pandas_query(sql)
