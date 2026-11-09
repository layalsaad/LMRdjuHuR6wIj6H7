import json
import logging

import numpy as np
import pandas as pd
import pytest

from term_deposit.config import STAGES, TARGET
from term_deposit.predict import load_model, predict, unseen_values
from term_deposit.predict import main as predict_main
from term_deposit.train import main as train_main
from term_deposit.train import metadata_path, model_path


@pytest.fixture(scope="module")
def pre_call_model(trained_dir):
    return load_model(model_path(trained_dir / "models", "pre_call"))


@pytest.mark.parametrize("stage", list(STAGES))
def test_train_saves_each_model_with_its_metadata(trained_dir, stage):
    path = model_path(trained_dir / "models", stage)
    metadata = json.loads(metadata_path(path).read_text())

    assert path.exists()
    assert metadata["features"] == STAGES[stage]
    assert metadata["training_rows"] == 600
    assert metadata["cross_validation"]["folds"] == 5
    assert 0 < metadata["threshold"] < 1
    assert 0 <= metadata["cross_validation"]["accuracy"]["mean"] <= 1
    assert len(metadata["data_sha256"]) == 64
    assert "student" in metadata["categories"]["job"]


def test_predict_adds_scores_and_keeps_other_columns(pre_call_model, calls_df):
    df = calls_df.drop(columns=["duration", TARGET]).assign(customer_id=range(len(calls_df)))

    scored = predict(pre_call_model, df)

    assert list(scored.columns) == [
        *df.columns,
        "probability_subscribe",
        "prediction",
        "priority_rank",
    ]
    assert scored["customer_id"].tolist() == df["customer_id"].tolist()
    assert scored["probability_subscribe"].between(0, 1).all()
    assert sorted(scored["priority_rank"]) == list(range(1, len(df) + 1))


def test_priority_rank_follows_the_probability(pre_call_model, calls_df):
    scored = predict(pre_call_model, calls_df)

    first = scored.loc[scored["priority_rank"] == 1, "probability_subscribe"].iloc[0]

    assert first == scored["probability_subscribe"].max()


def test_prediction_uses_the_learned_cut_off(pre_call_model, calls_df):
    scored = predict(pre_call_model, calls_df)
    above = scored["probability_subscribe"] >= pre_call_model.best_threshold_ + 1e-3
    below = scored["probability_subscribe"] < pre_call_model.best_threshold_ - 1e-3

    assert (scored.loc[above, "prediction"] == 1).all()
    assert (scored.loc[below, "prediction"] == 0).all()


def test_predict_ignores_an_empty_target_column(pre_call_model, calls_df):
    df = calls_df.assign(**{TARGET: np.nan})

    assert len(predict(pre_call_model, df)) == len(df)


def test_predict_rejects_invalid_customers(pre_call_model, calls_df):
    calls_df.loc[0, "age"] = 7

    with pytest.raises(ValueError, match="age"):
        predict(pre_call_model, calls_df)


def test_unseen_values_lists_only_new_categories():
    df = pd.DataFrame({"job": ["admin", "astronaut"], "month": ["may", "may"]})

    unseen = unseen_values(df, {"job": ["admin"], "month": ["may"]})

    assert unseen == {"job": ["astronaut"]}


def test_load_model_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="term-deposit-train"):
        load_model(tmp_path / "missing.joblib")


def test_predict_command_reports_quality_on_labelled_rows(trained_dir, tmp_path, caplog):
    output = tmp_path / "predictions.csv"
    model = model_path(trained_dir / "models", "post_call")

    with caplog.at_level(logging.INFO):
        predict_main(
            [str(trained_dir / "calls.csv"), "--model", str(model), "--output", str(output)]
        )

    assert len(pd.read_csv(output)) == 600
    assert "precision" in caplog.text


def test_predict_command_warns_about_new_categories(trained_dir, calls_df, tmp_path, caplog):
    new = tmp_path / "new.csv"
    calls_df.assign(job="astronaut").to_csv(new, index=False)
    model = model_path(trained_dir / "models", "pre_call")

    with caplog.at_level(logging.INFO):
        predict_main([str(new), "--model", str(model), "--output", str(tmp_path / "out.csv")])

    assert "astronaut" in caplog.text


def test_predict_command_exits_cleanly_on_invalid_input(trained_dir, tmp_path, capsys):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"age": [7]}).to_csv(bad, index=False)
    model = model_path(trained_dir / "models", "pre_call")

    with pytest.raises(SystemExit) as exit_info:
        predict_main([str(bad), "--model", str(model)])

    assert exit_info.value.code == 1
    assert "error: Missing required columns" in capsys.readouterr().err


def test_train_command_exits_cleanly_when_data_is_missing(tmp_path, capsys):
    with pytest.raises(SystemExit) as exit_info:
        train_main(["--data", str(tmp_path / "missing.csv")])

    assert exit_info.value.code == 1
    assert "error: Data file not found" in capsys.readouterr().err


def test_train_command_trains_one_stage(calls_csv, tmp_path, caplog):
    with caplog.at_level(logging.INFO):
        train_main(["--data", str(calls_csv), "--output-dir", str(tmp_path), "--stage", "pre_call"])

    assert model_path(tmp_path, "pre_call").exists()
    assert not model_path(tmp_path, "post_call").exists()
    assert "5-fold accuracy" in caplog.text