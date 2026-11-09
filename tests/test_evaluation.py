import numpy as np
import pandas as pd
import pytest
from scipy.stats import ttest_rel
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold

from term_deposit.config import STAGES, TARGET
from term_deposit.data import validate
from term_deposit.evaluation import (
    SCORING,
    corrected_ttest,
    cv_scores,
    gains_table,
    outer_cv,
    summarize,
    time_ordered_split,
)


def test_outer_cv_keeps_the_share_of_buyers_in_every_fold(calls_df):
    y = validate(calls_df)[TARGET]

    shares = [y.iloc[test].mean() for _, test in outer_cv().split(calls_df, y)]

    assert len(shares) == 5
    assert max(shares) - min(shares) < 0.01


def test_time_ordered_split_tests_on_the_last_rows():
    ((train, test),) = time_ordered_split(100, test_share=0.2)

    assert list(train) == list(range(80))
    assert list(test) == list(range(80, 100))


def test_cv_scores_one_row_per_fold_and_one_column_per_metric(calls_df):
    df = validate(calls_df)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

    scores = cv_scores(DummyClassifier(), df[STAGES["pre_call"]], df[TARGET], cv=cv, n_jobs=1)

    assert list(scores.columns) == list(SCORING)
    assert len(scores) == 3


def test_always_no_is_accurate_but_finds_no_buyers(calls_df):
    df = validate(calls_df)

    scores = cv_scores(DummyClassifier(), df[STAGES["pre_call"]], df[TARGET], n_jobs=1).mean()

    assert scores["accuracy"] == pytest.approx(1 - df[TARGET].mean(), abs=0.01)
    assert scores["recall"] == 0
    assert scores["precision"] == 0
    assert scores["roc_auc"] == 0.5


def test_summarize_puts_the_best_average_precision_first():
    scores = {
        "weak": pd.DataFrame({"average_precision": [0.1, 0.2], "accuracy": [0.9, 0.9]}),
        "strong": pd.DataFrame({"average_precision": [0.5, 0.4], "accuracy": [0.8, 0.8]}),
    }

    summary = summarize(scores)

    assert list(summary.index) == ["strong", "weak"]
    assert summary.loc["strong", "average_precision_mean"] == pytest.approx(0.45)


def test_corrected_ttest_detects_a_consistent_improvement():
    rng = np.random.default_rng(0)
    baseline = rng.normal(0.25, 0.02, 5)

    gain, p_value = corrected_ttest(baseline + 0.1, baseline + rng.normal(0, 0.005, 5))

    assert gain == pytest.approx(0.1, abs=0.01)
    assert p_value < 0.01


def test_corrected_ttest_is_more_cautious_than_a_plain_ttest():
    rng = np.random.default_rng(1)
    a, b = rng.normal(0.26, 0.02, 5), rng.normal(0.25, 0.02, 5)

    _, p_value = corrected_ttest(a, b)

    assert p_value > ttest_rel(a, b).pvalue


def test_corrected_ttest_identical_scores():
    scores = np.full(5, 0.25)

    assert corrected_ttest(scores, scores) == (0.0, 1.0)


def test_gains_table_perfect_ranking_finds_every_buyer_first():
    y = np.array([1] * 10 + [0] * 90)
    scores = np.linspace(1, 0, 100)

    table = gains_table(y, scores)

    assert table.loc[0, "buyers_found"] == 10
    assert table.loc[0, "gains"] == 1.0
    assert table.loc[0, "lift"] == pytest.approx(10)
    assert table["gains"].iloc[-1] == 1.0


def test_gains_table_equal_scores_give_no_lift():
    y = np.array([1] * 100 + [0] * 900)

    table = gains_table(y, np.zeros(1000))

    assert table["lift"].between(0.7, 1.3).all()