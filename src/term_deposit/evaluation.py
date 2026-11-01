"""How models are graded: the cross-validation splits, the metrics, significance and gains.

These rules are fixed before any model is built, so they cannot be bent to flatter one.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score, make_scorer, precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate

from term_deposit.config import CV_SPLITS, RANDOM_STATE

# Accuracy because the brief asks for it; the others measure how well buyers are found.
# zero_division=0: a model that predicts no buyers scores 0 instead of raising a warning.
SCORING = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": "recall",
    "f1": make_scorer(f1_score, zero_division=0),
}
# Models are ranked, and tuned, on average precision (PR-AUC): it rewards putting buyers at the
# top of the call list, where the bank actually calls.
MAIN_METRIC = "average_precision"


def outer_cv(random_state: int = RANDOM_STATE) -> StratifiedKFold:
    """The brief's 5 folds. Stratified: every fold keeps the overall share of buyers."""
    return StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=random_state)


def inner_cv(random_state: int = RANDOM_STATE) -> StratifiedKFold:
    """Folds for tuning a model inside each outer training fold, so tuning never sees the
    outer test fold. Three folds keep tuning affordable on 32,000 training rows."""
    return StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)


def time_ordered_split(n_rows: int, test_share: float = 0.2) -> list[tuple[np.ndarray, ...]]:
    """One split that trains on the earlier rows and tests on the later ones.

    The rows are in date order, so this mimics using the model on a future campaign.
    Pass it as `cv` to cv_scores.
    """
    cut = int(n_rows * (1 - test_share))
    return [(np.arange(cut), np.arange(cut, n_rows))]


def cv_scores(
    estimator: BaseEstimator, X: pd.DataFrame, y: pd.Series, cv=None, n_jobs: int | None = -1
) -> pd.DataFrame:
    """Score `estimator` on test folds it never saw, one row per fold and one column per metric.

    If the estimator tunes itself (a GridSearchCV), tuning happens inside each training fold,
    so the scores are not inflated by tuning on the test data (nested cross-validation).
    """
    results = cross_validate(
        estimator,
        X,
        y,
        cv=cv if cv is not None else outer_cv(),
        scoring=SCORING,
        n_jobs=n_jobs,
    )
    return pd.DataFrame({metric: results[f"test_{metric}"] for metric in SCORING})


def summarize(scores: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Mean and standard deviation of every metric per model, best MAIN_METRIC first."""
    summary = pd.concat(
        {name: df.agg(["mean", "std"]).unstack() for name, df in scores.items()}, axis=1
    ).T
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.sort_values(f"{MAIN_METRIC}_mean", ascending=False)


def corrected_ttest(
    scores_a: np.ndarray, scores_b: np.ndarray, n_splits: int = CV_SPLITS
) -> tuple[float, float]:
    """Is model a really better than model b, or did it get lucky on these folds?

    Uses the Nadeau & Bengio correction: CV training sets overlap, so a plain t-test on fold
    scores is overconfident. Returns (mean difference a - b, two-sided p-value).
    """
    diff = np.asarray(scores_a) - np.asarray(scores_b)
    n = len(diff)
    mean, var = diff.mean(), diff.var(ddof=1)
    if var == 0:
        return float(mean), 1.0 if mean == 0 else 0.0
    test_train_ratio = 1 / (n_splits - 1)
    t_stat = mean / np.sqrt(var * (1 / n + test_train_ratio))
    p_value = 2 * stats.t.sf(abs(t_stat), df=n - 1)
    return float(mean), float(p_value)


def out_of_fold_scores(
    estimator: BaseEstimator, X: pd.DataFrame, y: pd.Series, cv=None, n_jobs: int | None = -1
) -> np.ndarray:
    """Each customer's probability of buying, from a model trained without that customer."""
    proba = cross_val_predict(
        estimator,
        X,
        y,
        cv=cv if cv is not None else outer_cv(),
        method="predict_proba",
        n_jobs=n_jobs,
    )
    return proba[:, 1]


def gains_table(y_true, scores, groups: int = 10, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """What calling only the top-scored customers achieves, in tenths of the customer list.

    Row k answers: if the bank calls the k tenths of customers with the highest scores, what
    share of all buyers does it reach (gains), what share of those calls succeed
    (success_rate), and how many times better than calling at random is that (lift)?
    Customers with equal scores are put in random order, so ties favour no one.
    """
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    shuffled = np.random.default_rng(random_state).permutation(len(y_true))
    order = shuffled[np.argsort(-scores[shuffled], kind="stable")]
    called = np.ceil(np.arange(1, groups + 1) / groups * len(y_true)).astype(int)
    found = np.cumsum(y_true[order])[called - 1]
    success_rate = found / called
    return pd.DataFrame(
        {
            "share_called": called / len(y_true),
            "customers_called": called,
            "buyers_found": found,
            "gains": found / y_true.sum(),
            "success_rate": success_rate,
            "lift": success_rate / y_true.mean(),
        }
    )