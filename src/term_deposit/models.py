"""Candidate models, how each one prepares the data, and the settings worth tuning."""

from dataclasses import dataclass, field

from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from term_deposit.config import BINARY, CATEGORICAL, NUMERIC, RANDOM_STATE, YES_NO
from term_deposit.evaluation import MAIN_METRIC, inner_cv


def make_preparer(features: list[str], scale: bool = False) -> ColumnTransformer:
    """Turn the raw columns in `features` into numbers a model can use.

    Categories become one 0/1 column per value (job_student, job_retired, ...); a value never
    seen in training gets all zeros instead of an error. yes/no columns become 1/0. Numbers
    are kept as they are for tree models, and scaled to a common range when `scale` is True,
    because a linear model compares their weights. The output is a plain (dense) table, which
    every model accepts.
    """
    numeric = [col for col in NUMERIC if col in features]
    categorical = [col for col in CATEGORICAL if col in features]
    binary = [col for col in BINARY if col in features]
    return ColumnTransformer(
        [
            ("numeric", StandardScaler() if scale else "passthrough", numeric),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
            ("yes_no", OrdinalEncoder(categories=[list(YES_NO)] * len(binary)), binary),
        ],
        verbose_feature_names_out=False,
    )


@dataclass(frozen=True)
class Candidate:
    """A model, the settings worth tuning, and whether it needs scaled numbers."""

    model: BaseEstimator
    param_grid: dict[str, list] = field(default_factory=dict)
    scale: bool = False

    def pipeline(self, features: list[str]) -> Pipeline:
        """Data preparation and model in one object, so both are fitted on training data only."""
        return Pipeline(
            [("prepare", make_preparer(features, self.scale)), ("model", clone(self.model))]
        )

    def tuned(self, features: list[str]) -> BaseEstimator:
        """A pipeline that picks its own settings with an inner CV when fitted."""
        if not self.param_grid:
            return self.pipeline(features)
        grid = {f"model__{name}": values for name, values in self.param_grid.items()}
        return GridSearchCV(self.pipeline(features), grid, cv=inner_cv(), scoring=MAIN_METRIC)


def get_candidates(random_state: int = RANDOM_STATE) -> dict[str, Candidate]:
    """Models compared in notebook 02, from simplest to most flexible.

    Grids are small on purpose: a few sensible values per setting keep nested CV affordable,
    and large searches mostly fit noise. No class weights: imbalance is handled by ranking on
    average precision and by tuning the decision threshold, which keeps probabilities honest.
    """
    return {
        "baseline": Candidate(DummyClassifier(strategy="most_frequent")),
        "logistic_regression": Candidate(
            LogisticRegression(max_iter=1000),
            {"C": [0.01, 0.1, 1.0]},
            scale=True,
        ),
        "decision_tree": Candidate(
            DecisionTreeClassifier(random_state=random_state),
            {"max_depth": [3, 5, 8], "min_samples_leaf": [50, 200]},
        ),
        "random_forest": Candidate(
            RandomForestClassifier(n_estimators=200, random_state=random_state),
            {"max_depth": [8, 16], "min_samples_leaf": [5, 20]},
        ),
        "gradient_boosting": Candidate(
            HistGradientBoostingClassifier(random_state=random_state),
            {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [15, 31]},
        ),
    }