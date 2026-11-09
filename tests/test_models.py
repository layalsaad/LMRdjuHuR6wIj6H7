import pytest
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GridSearchCV, cross_val_predict

from term_deposit.config import STAGES, TARGET
from term_deposit.data import validate
from term_deposit.models import get_candidates, make_preparer, with_threshold


@pytest.fixture
def pre_call(calls_df):
    df = validate(calls_df)
    return df[STAGES["pre_call"]], df[TARGET]


def test_preparer_turns_columns_into_numbers(pre_call):
    X, _ = pre_call

    prepared = make_preparer(STAGES["pre_call"]).fit(X)
    names = list(prepared.get_feature_names_out())

    assert "duration" not in names
    assert {"age", "job_student", "housing"} <= set(names)
    assert prepared.transform(X).shape == (len(X), len(names))


def test_preparer_scores_unseen_categories_instead_of_failing(pre_call):
    X, _ = pre_call
    prepared = make_preparer(STAGES["pre_call"]).fit(X)

    row = prepared.transform(X.head(1).assign(job="astronaut"))
    job_columns = [i for i, name in enumerate(prepared.get_feature_names_out()) if "job_" in name]

    assert row[0, job_columns].sum() == 0


@pytest.mark.parametrize("name", list(get_candidates()))
def test_every_candidate_fits_and_gives_probabilities(name, pre_call):
    X, y = pre_call

    model = get_candidates()[name].pipeline(STAGES["pre_call"]).fit(X, y)
    proba = model.predict_proba(X)

    assert proba.shape == (len(X), 2)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_tuned_searches_settings_only_when_there_is_a_grid():
    candidates = get_candidates()

    assert isinstance(candidates["gradient_boosting"].tuned(STAGES["pre_call"]), GridSearchCV)
    assert not isinstance(candidates["baseline"].tuned(STAGES["pre_call"]), GridSearchCV)


def test_gradient_boosting_finds_buyers_better_than_guessing(pre_call):
    X, y = pre_call
    model = get_candidates()["gradient_boosting"].pipeline(STAGES["pre_call"])

    proba = cross_val_predict(model, X, y, cv=3, method="predict_proba")[:, 1]

    assert average_precision_score(y, proba) > 2 * y.mean()


def test_with_threshold_lowers_the_cut_off_for_rare_buyers(pre_call):
    X, y = pre_call

    model = with_threshold(get_candidates()["logistic_regression"].pipeline(STAGES["pre_call"]))
    model.fit(X, y)

    assert 0 < model.best_threshold_ < 0.5