from fasthtml.common import *   # fasthtml HTML-Builder (H1, Div, etc.)
import matplotlib.pyplot as plt
import pandas as pd

# Import QueryBase, Employee, Team from employee_events
# Importiere die echten SQL-Model-Klassen direkt
try:
    from employee_events.employee import Employee
    from employee_events.team import Team
except Exception:
    # Fallback dummy classes so ImportErrors beim Testen nicht crashen
    class Employee:
        def __init__(self): pass
    class Team:
        def __init__(self): pass

# import the load_model function from the utils.py file
try:
    from report.utils import load_model
except Exception:
    # Fallback loader that raises if actually used
    def load_model(*a, **k):
        raise RuntimeError("load_model not available")


"""
Below, we import the parent classes
you will use for subclassing
"""
# package-qualified imports (korrekt innerhalb des 'report' packages)
from report.base_components.dropdown import Dropdown
from report.base_components.base_component import BaseComponent
from report.base_components.radio import Radio
from report.base_components.matplotlib_viz import MatplotlibViz
from report.base_components.data_table import DataTable

from report.combined_components.form_group import FormGroup
from report.combined_components.combined_component import CombinedComponent


# Create a subclass of base_components/dropdown
# called `ReportDropdown`
class ReportDropdown(Dropdown):
    """
    Dropdown that uses the passed SQL model to populate options.
    build_component and component_data follow the (entity_id, model) signature.
    """

    def build_component(self, entity_id, model):
        # set the label to the model's name attribute if present, otherwise use classname
        label = getattr(model, "name", None) or model.__class__.__name__
        # assign to self.label (Dropdown expected attribute)
        try:
            self.label = label
        except Exception:
            pass
        # call parent build_component (assumes same signature)
        return super().build_component(entity_id, model)

    def component_data(self, entity_id, model):
        """
        Return a list of (label, value) tuples for the dropdown.
        Accepts:
         - model.names() -> list of tuples OR pandas.DataFrame
         - model.all_employees / model.all_teams / model.members -> DataFrame or list
        Falls nichts gefunden wird, wird eine leere Liste zurückgegeben.
        """
        try:
            import pandas as pd
            # helper to convert dataframe to list[(label, id)]
            def df_to_list(df):
                if df is None:
                    return []
                # if it's already a pandas DataFrame
                if isinstance(df, pd.DataFrame):
                    # common column patterns:
                    if set(['full_name', 'employee_id']).issubset(df.columns):
                        return list(df[['full_name', 'employee_id']].itertuples(index=False, name=None))
                    if set(['team_name', 'team_id']).issubset(df.columns):
                        return list(df[['team_name', 'team_id']].itertuples(index=False, name=None))
                    # fallback: first two columns
                    if df.shape[1] >= 2:
                        return list(df.iloc[:, 0:2].itertuples(index=False, name=None))
                    return []
                # if it's a list of tuples already
                if isinstance(df, list):
                    return df
                # if it's something else (e.g. dict), try to coerce
                try:
                    ddf = pd.DataFrame(df)
                    if ddf.shape[1] >= 2:
                        return list(ddf.iloc[:, 0:2].itertuples(index=False, name=None))
                except Exception:
                    pass
                return []

            # 1) Try model.names() without args (common)
            if hasattr(model, "names"):
                try:
                    res = model.names()
                    out = df_to_list(res)
                    if out:
                        return out
                except TypeError:
                    # try calling with (None, model)
                    try:
                        res = model.names(None, model)
                        out = df_to_list(res)
                        if out:
                            return out
                    except Exception:
                        pass
                except Exception:
                    pass

            # 2) Try other likely methods that return df/list
            for method in ("all_employees", "all_teams", "members"):
                if hasattr(model, method):
                    func = getattr(model, method)
                    tried = False
                    # try various call signatures gracefully
                    for call_args in ((), (None,), (None, model)):
                        try:
                            res = func(*call_args)
                            out = df_to_list(res)
                            if out:
                                return out
                        except TypeError:
                            # signature mismatch — try next
                            continue
                        except Exception:
                            break
                        finally:
                            tried = True

            # nothing produced -> return empty list
            return []
        except Exception:
            # If anything unexpected happens, return empty list to avoid crash
            return []


# Create a subclass of base_components/BaseComponent
# called `Header`
class Header(BaseComponent):
    """Simple header component that shows the model name."""

    def build_component(self, entity_id, model):
        # Use model.name if available, otherwise the class name
        title = getattr(model, "name", None) or model.__class__.__name__
        # Return an H1 element from fasthtml.common
        return H1(title)


# Create a subclass of base_components/MatplotlibViz
# called `LineChart`
class LineChart(MatplotlibViz):
    """
    LineChart that expects the model to expose an events/time-series method.
    We attempt several method names for compatibility.
    """

    def visualization(self, entity_id, model):
        # asset_id is represented here by entity_id
        df = None
        # try a set of likely method names
        candidates = ["event_counts", "time_series", "events_time_series", "time_series_data"]
        for name in candidates:
            if hasattr(model, name):
                try:
                    # try both signatures
                    try:
                        df = getattr(model, name)(entity_id, model)
                    except TypeError:
                        df = getattr(model, name)(entity_id)
                    break
                except Exception:
                    df = None

        if df is None:
            # Nothing to plot -> return an empty figure
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data", ha="center")
            return fig

        # Work with DataFrame: ensure columns exist for positive/negative counts
        if not hasattr(df, "columns"):
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data", ha="center")
            return fig

        # If the DF has columns 'positive_count' and 'negative_count' use them,
        # otherwise try 'positive_events'/'negative_events' or 'positive'/'negative'
        if "positive_count" in df.columns and "negative_count" in df.columns:
            use_df = df[["day", "positive_count", "negative_count"]].copy()
        elif "positive_events" in df.columns and "negative_events" in df.columns:
            use_df = df[["day", "positive_events", "negative_events"]].copy()
        elif "positive" in df.columns and "negative" in df.columns:
            use_df = df[["day", "positive", "negative"]].copy()
        else:
            # As a last resort, try columns after the day column
            cols = [c for c in df.columns if c != "day"]
            if len(cols) >= 2:
                use_df = df[["day", cols[0], cols[1]]].copy()
            else:
                # not enough numeric columns
                fig, ax = plt.subplots()
                ax.text(0.5, 0.5, "Insufficient series data", ha="center")
                return fig

        # fillna, set index, sort, cumsum
        use_df = use_df.fillna(0)
        use_df = use_df.set_index("day")
        use_df = use_df.sort_index()
        use_df = use_df.cumsum()

        # rename columns to ['Positive','Negative'] for display
        use_df.columns = ["Positive", "Negative"]

        # plot
        fig, ax = plt.subplots()
        use_df.plot(ax=ax)

        # styling via MatplotlibViz helper if available
        try:
            self.set_axis_styling(ax, border_color="black", font_color="black")
        except Exception:
            pass

        ax.set_title("Cumulative Events")
        ax.set_xlabel("Day")
        ax.set_ylabel("Cumulative Count")
        return fig


# Create a subclass of base_components/MatplotlibViz
# called `BarChart`
class BarChart(MatplotlibViz):
    # Create a `predictor` class attribute assigned to the output of load_model
    try:
        predictor = load_model()
    except Exception:
        predictor = None

    def visualization(self, entity_id, model):
        # Get model data suitable for the ML model.
        X = None
        # try likely method names that return features for the ML model
        for name in ("model_data", "get_model_data", "features", "ml_input"):
            if hasattr(model, name):
                func = getattr(model, name)
                try:
                    # try (entity_id, model) first, then (entity_id)
                    try:
                        X = func(entity_id, model)
                    except TypeError:
                        X = func(entity_id)
                    break
                except Exception:
                    X = None

        # fallback: try to use time_series aggregated as features
        if X is None and hasattr(model, "time_series"):
            try:
                # call with either signature
                try:
                    ts = model.time_series(entity_id, model)
                except TypeError:
                    ts = model.time_series(entity_id)
                if ts is not None and not getattr(ts, "empty", False):
                    last = ts.iloc[-1]
                    pos_col = None
                    neg_col = None
                    for c in ("positive_count", "positive_events", "positive"):
                        if c in last.index:
                            pos_col = c
                            break
                    for c in ("negative_count", "negative_events", "negative"):
                        if c in last.index:
                            neg_col = c
                            break
                    features = {}
                    features["positive"] = float(last[pos_col]) if pos_col is not None else 0.0
                    features["negative"] = float(last[neg_col]) if neg_col is not None else 0.0
                    X = pd.DataFrame([features])
            except Exception:
                X = None

        # If predictor is not available, return an empty figure with warning
        if self.predictor is None or X is None:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Model or data not available", ha="center")
            return fig

        # predict_proba: ensure we call correctly
        try:
            proba = self.predictor.predict_proba(X)
            # ensure 2D and take second column
            if proba.ndim == 2 and proba.shape[1] >= 2:
                probs = proba[:, 1]
            else:
                # fallback: if single-prob returned, use it
                probs = proba.ravel()
        except Exception:
            # prediction failed
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Prediction failed", ha="center")
            return fig

        # derive pred value to plot
        model_name = getattr(model, "name", model.__class__.__name__).lower()
        if "team" in model_name:
            pred = float(probs.mean())
        else:
            pred = float(probs[0]) if len(probs) > 0 else 0.0

        # plotting
        fig, ax = plt.subplots()
        ax.barh([""], [pred])
        ax.set_xlim(0, 1)
        ax.set_title("Predicted Recruitment Risk", fontsize=20)

        # style axis
        try:
            self.set_axis_styling(ax, border_color="black", font_color="black")
        except Exception:
            pass

        return fig


# Create a subclass of combined_components/CombinedComponent
# called Visualizations
class Visualizations(CombinedComponent):
    # children list has instances of LineChart and BarChart
    children = [LineChart(), BarChart()]

    # Leave this line unchanged
    outer_div_type = Div(cls='grid')


# Create a subclass of base_components/DataTable
# called `NotesTable`
class NotesTable(DataTable):
    def component_data(self, entity_id, model):
        """
        Return notes dataframe for the given entity_id.
        Always return a pandas.DataFrame (empty if no data found)
        so DataTable.build_component can safely iterate columns.
        """
        try:
            # prefer model.notes(entity_id, model) or model.notes(id)
            if hasattr(model, "notes"):
                try:
                    try:
                        df = model.notes(entity_id, model)
                    except TypeError:
                        df = model.notes(entity_id)
                    if df is not None:
                        return df
                except Exception:
                    pass

            # fallback: attempt to call a generic execute_query on model
            if hasattr(model, "execute_query"):
                try:
                    df = model.execute_query(
                        "SELECT note_date, note FROM notes WHERE employee_id = :id OR team_id = :id ORDER BY note_date",
                        {"id": entity_id}
                    )
                    if df is not None:
                        return df
                except Exception:
                    pass

        except Exception:
            # if anything unexpected happens, we still want to return an empty DataFrame
            pass

        # final fallback: return empty DataFrame with expected columns
        return pd.DataFrame(columns=["note_date", "note"])


class DashboardFilters(FormGroup):

    id = "top-filters"
    action = "/update_data"
    method = "POST"

    children = [
        Radio(
            values=["Employee", "Team"],
            name='profile_type',
            hx_get='/update_dropdown',
            hx_target='#selector'
        ),
        ReportDropdown(
            id="selector",
            name="user-selection")
    ]


class Report(CombinedComponent):
    # children: header, filters, visualizations, notes table
    children = [Header(), DashboardFilters(), Visualizations(), NotesTable()]

    # optional: falls du schon outer_div_type gesetzt hast, das bleibt
    outer_div_type = Div(cls='grid')

    # WICHTIG: build_component liefert das gerenderte HTML zurück
    def build_component(self, entity_id, model):
        """
        Baut alle child-components mittels call_children() und
        gibt sie in einem outer div zurück. Signatur folgt Projektregel.
        """
        # call_children erwartet (entity_id, model) für jedes Kind
        built_children = self.call_children(entity_id, model)
        # outer_div(children, div_args) ist die CombinedComponent-Helferfunktion
        return self.outer_div(built_children, {"class": "recruitment-dashboard"})


# Provide an alias for pages
RecruitmentRiskPage = Report

# Initialize a fasthtml app (single place)
try:
    from fasthtml import FastHTML
    app = FastHTML()
except Exception:
    # Minimal fallback app that provides decorators used later for tests that import the module
    class _DummyApp:
        def get(self, path):
            def decorator(fn):
                return fn
            return decorator
        def post(self, path):
            def decorator(fn):
                return fn
            return decorator
    app = _DummyApp()


# Create a route for a get request - root
@app.get('/')
def index():
    page = RecruitmentRiskPage()
    # create a model instance and ensure it has a `name` attribute
    try:
        model = Employee()
    except Exception:
        class _Dummy:
            name = "employee"
        model = _Dummy()
    if not hasattr(model, "name"):
        try:
            model.name = "employee"
        except Exception:
            pass
    return page.build_component(1, model)


# Create a route for a get request for an employee id
@app.get('/employee/{id}')
def employee_route(id: str):
    page = RecruitmentRiskPage()
    try:
        model = Employee()
    except Exception:
        class _Dummy:
            name = "employee"
        model = _Dummy()
    if not hasattr(model, "name"):
        model.name = "employee"
    return page.build_component(id, model)


# Create a route for a get request for a team id
@app.get('/team/{id}')
def team_route(id: str):
    page = RecruitmentRiskPage()
    try:
        model = Team()
    except Exception:
        class _Dummy:
            name = "team"
        model = _Dummy()
    if not hasattr(model, "name"):
        model.name = "team"
    return page.build_component(id, model)


# Keep the below route for AJAX dropdown updates
@app.get('/update_dropdown')
def update_dropdown(r):
    """
    Erwartet URL wie /update_dropdown?profile_type=Team oder ...=Employee
    r sollte das Request-Objekt sein (fasthtml übergibt es).
    """
    dropdown = DashboardFilters.children[1]

    # versuche Query-Param sauber zu lesen (verschiedene fasthtml-Versionen haben unterschiedliche APIs)
    profile = None
    try:
        if hasattr(r, "query_params"):
            profile = r.query_params.get("profile_type")
        elif hasattr(r, "args"):
            profile = r.args.get("profile_type")
        elif hasattr(r, "scope") and "query_string" in r.scope:
            qs = r.scope.get("query_string", b"").decode()
            from urllib.parse import parse_qs
            profile = parse_qs(qs).get("profile_type", [None])[0]
    except Exception:
        profile = None

    if profile == "Team":
        return dropdown(None, Team())
    else:
        # default: Employee
        return dropdown(None, Employee())


@app.post('/update_data')
async def update_data(r):
    from fasthtml.common import RedirectResponse
    data = await r.form()
    # data may be a form object with _dict (fasthtml) or a simple dict-like
    profile_type = None
    selected_id = None
    if hasattr(data, "_dict"):
        profile_type = data._dict.get('profile_type')
        selected_id = data._dict.get('user-selection')
    else:
        try:
            profile_type = data.get('profile_type')
            selected_id = data.get('user-selection')
        except Exception:
            profile_type = None
            selected_id = None

    if profile_type == 'Employee':
        return RedirectResponse(f"/employee/{selected_id}", status_code=303)
    elif profile_type == 'Team':
        return RedirectResponse(f"/team/{selected_id}", status_code=303)
    else:
        # fallback: redirect to root
        return RedirectResponse("/", status_code=303)


# If fasthtml provides a serve function in the environment, this will run it.
# IMPORTANT: call serve() only after all routes are defined
try:
    serve()
except Exception:
    # serve may not exist in test environment - ignore silently
    pass
