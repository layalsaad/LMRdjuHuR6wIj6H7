import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from term_deposit.config import PROFILE_FEATURES, STAGES, TARGET
from term_deposit.data import validate
from term_deposit.insights import (
    _condition,
    permutation_importance_cv,
    segment_names,
    segment_tree,
    subscription_rates,
)
from term_deposit.models import get_candidates


def test_subscription_rates_with_intervals(calls_df):
    df = validate(calls_df)

    table = subscription_rates(df, "housing")

    assert table["customers"].sum() == len(df)
    assert ((table["low"] <= table["rate"]) & (table["rate"] <= table["high"])).all()
    assert table["rate"].is_monotonic_decreasing
    assert table.index[0] == "no"  # buyers in the fake data more often have no housing loan


def test_small_groups_get_wider_intervals():
    df = pd.DataFrame({"group": ["small"] * 10 + ["big"] * 1000, TARGET: [1, 0] * 505})

    table = subscription_rates(df, "group")
    width = table["high"] - table["low"]

    assert width["small"] > width["big"]


def test_permutation_importance_finds_the_column_that_drives_buying(calls_df):
    df = validate(calls_df)
    features = STAGES["post_call"]
    model = get_candidates()["logistic_regression"].pipeline(features)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

    importance = permutation_importance_cv(model, df[features], df[TARGET], cv=cv, n_jobs=1)

    assert importance.shape == (3, len(features))
    assert importance.mean().idxmax() == "duration"


@pytest.mark.parametrize(
    ("column", "threshold", "above", "text"),
    [
        ("age", 56.5, False, "age ≤ 56"),
        ("age", 56.5, True, "age ≥ 57"),
        ("balance", 1161.5, True, "balance ≥ 1,162"),
        ("housing", 0.5, False, "housing: no"),
        ("job_student", 0.5, True, "job is student"),
        ("job_student", 0.5, False, "job is not student"),
    ],
)
def test_conditions_read_like_sentences(column, threshold, above, text):
    is_category = {"job_student": ("job", "student")}

    assert _condition(column, threshold, is_category, above) == text


def test_segments_are_few_large_and_readable(calls_df):
    df = validate(calls_df)
    X = df[PROFILE_FEATURES]

    segments = segment_tree(PROFILE_FEATURES, max_depth=2, min_share=0.1).fit(X, df[TARGET])
    names = segment_names(segments, X)

    assert names.notna().all()
    assert names.nunique() <= 4
    assert names.value_counts(normalize=True).min() >= 0.1
    assert all(" and " in name for name in names.unique())