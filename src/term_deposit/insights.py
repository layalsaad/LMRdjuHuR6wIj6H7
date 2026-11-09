"""Turning the data and the models into business answers: who subscribes, and why."""

import math

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, clone
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.parallel import Parallel, delayed

from term_deposit.config import BINARY, RANDOM_STATE, TARGET
from term_deposit.evaluation import MAIN_METRIC, outer_cv
from term_deposit.models import make_preparer


def subscription_rates(
    df: pd.DataFrame, column: str, sort: bool = True, target: str = TARGET
) -> pd.DataFrame:
    """Customers, subscribers and subscription rate for each value of `column`.

    `low` and `high` bound a 95% Wilson interval: the range the true rate plausibly lies in,
    given how many customers the group has. Small groups get wide intervals, so a striking
    rate from a handful of customers is not mistaken for a pattern. Sorted from the highest
    rate, unless `sort` is False (for groups with a natural order, such as age bands).
    """
    table = df.groupby(column, observed=True)[target].agg(customers="size", subscribed="sum")
    intervals = [
        stats.binomtest(int(k), int(n)).proportion_ci(method="wilson")
        for k, n in zip(table["subscribed"], table["customers"], strict=True)
    ]
    table = table.assign(
        rate=table["subscribed"] / table["customers"],
        low=[ci.low for ci in intervals],
        high=[ci.high for ci in intervals],
    )
    return table.sort_values("rate", ascending=False) if sort else table


def _fold_importance(estimator, X, y, train, test, n_repeats, random_state) -> np.ndarray:
    model = clone(estimator).fit(X.iloc[train], y.iloc[train])
    result = permutation_importance(
        model,
        X.iloc[test],
        y.iloc[test],
        scoring=MAIN_METRIC,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    return result.importances_mean


def permutation_importance_cv(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv=None,
    n_repeats: int = 5,
    random_state: int = RANDOM_STATE,
    n_jobs: int | None = -1,
) -> pd.DataFrame:
    """How much worse the model ranks buyers when one column is shuffled.

    The model is trained on each training fold and probed on the matching test fold, so this
    measures what each column contributes to predicting customers it has not seen. Raw columns
    are shuffled, so a category such as job counts as one column, not twelve. Returns one row
    per fold and one column per input column: the drop in MAIN_METRIC.
    """
    cv = cv if cv is not None else outer_cv()
    rows = Parallel(n_jobs=n_jobs)(
        delayed(_fold_importance)(estimator, X, y, train, test, n_repeats, random_state)
        for train, test in cv.split(X, y)
    )
    return pd.DataFrame(rows, columns=X.columns)


def segment_tree(features: list[str], max_depth: int = 3, min_share: float = 0.02) -> Pipeline:
    """A shallow decision tree that splits customers into at most 2**max_depth large groups.

    Every group holds at least `min_share` of the customers it is trained on, so each one is big
    enough to plan a campaign around and its rate is not a fluke of a few customers.
    """
    return Pipeline(
        [
            ("prepare", make_preparer(features)),
            (
                "tree",
                DecisionTreeClassifier(
                    max_depth=max_depth, min_samples_leaf=min_share, random_state=RANDOM_STATE
                ),
            ),
        ]
    )


def _condition(column: str, threshold: float, is_category: dict, above: bool) -> str:
    """One step of a tree path as text, e.g. "marital is single" or "age ≥ 57"."""
    if column in is_category:
        name, value = is_category[column]
        return f"{name} is {value}" if above else f"{name} is not {value}"
    if column in BINARY:
        return f"{column}: yes" if above else f"{column}: no"
    # The numeric columns hold whole numbers, so "≤ 616.5" means "≤ 616" and "> 616.5" "≥ 617".
    cut = math.floor(threshold)
    return f"{column} ≥ {cut + 1:,}" if above else f"{column} ≤ {cut:,}"


def segment_names(fitted: Pipeline, X: pd.DataFrame) -> pd.Series:
    """The readable rule of the segment each customer falls in, e.g.
    "marital is not married and balance ≥ 617 and housing: no"."""
    prepare, tree = fitted["prepare"], fitted["tree"].tree_
    encoder = prepare.named_transformers_["categorical"]
    is_category = {
        f"{column}_{value}": (column, value)
        for column, values in zip(encoder.feature_names_in_, encoder.categories_, strict=True)
        for value in values
    }
    names = prepare.get_feature_names_out()

    rules = {}
    stack = [(0, [])]
    while stack:
        node, conditions = stack.pop()
        left, right = tree.children_left[node], tree.children_right[node]
        if left == right:  # a leaf: both children are -1
            rules[node] = " and ".join(conditions) or "all customers"
            continue
        column, threshold = names[tree.feature[node]], tree.threshold[node]
        stack.append((left, [*conditions, _condition(column, threshold, is_category, False)]))
        stack.append((right, [*conditions, _condition(column, threshold, is_category, True)]))

    leaves = fitted["tree"].apply(prepare.transform(X))
    return pd.Series([rules[leaf] for leaf in leaves], index=X.index, name="segment")