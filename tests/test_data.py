import pytest

from term_deposit.config import FEATURES, STAGES, TARGET
from term_deposit.data import load_data, validate


def test_load_data_returns_target_as_zero_and_one(calls_csv, calls_df):
    df = load_data(calls_csv)

    assert list(df.columns) == [*FEATURES, TARGET]
    assert set(df[TARGET]) == {0, 1}
    assert df[TARGET].sum() == (calls_df[TARGET] == "yes").sum()


def test_load_data_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Data file not found"):
        load_data(tmp_path / "missing.csv")


def test_validate_reports_missing_columns(calls_df):
    with pytest.raises(ValueError, match=r"Missing required columns: \['loan'\]"):
        validate(calls_df.drop(columns="loan"))


def test_validate_lists_every_problem_at_once(calls_df):
    calls_df.loc[0, "age"] = 7
    calls_df.loc[1, "housing"] = "Yes"
    calls_df.loc[2, "month"] = None
    calls_df["campaign"] = calls_df["campaign"] + 0.5

    with pytest.raises(ValueError) as error:
        validate(calls_df)

    message = str(error.value)
    assert "missing values in ['month']" in message
    assert "non-integer values in ['campaign']" in message
    assert "age (18, 100)" in message
    assert "values other than 'yes'/'no' in ['housing']" in message


def test_validate_rejects_a_target_other_than_yes_or_no(calls_df):
    calls_df.loc[0, TARGET] = "maybe"

    with pytest.raises(ValueError, match=r"'yes'/'no' in \['y'\]"):
        validate(calls_df)


def test_validate_new_customers_need_no_duration_or_target(calls_df):
    features = STAGES["pre_call"]
    new_customers = calls_df[features].assign(customer_id=range(len(calls_df)))

    checked = validate(new_customers, require_target=False, features=features)

    assert list(checked.columns) == features


def test_validate_keeps_unknown_categories(calls_df):
    calls_df.loc[0, "job"] = "astronaut"

    assert validate(calls_df).loc[0, "job"] == "astronaut"