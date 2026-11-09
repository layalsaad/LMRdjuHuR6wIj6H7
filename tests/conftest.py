import numpy as np
import pandas as pd
import pytest

from term_deposit.train import train


def make_calls(n_rows: int = 600, seed: int = 0) -> pd.DataFrame:
    """Synthetic call data with the real schema. Buyers are more often over 60, have no housing
    loan and had longer calls, so a working model can find them."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "age": rng.integers(19, 90, n_rows),
            "job": rng.choice(["admin", "blue-collar", "retired", "student"], n_rows),
            "marital": rng.choice(["married", "single", "divorced"], n_rows),
            "education": rng.choice(["primary", "secondary", "tertiary", "unknown"], n_rows),
            "default": rng.choice(["no", "yes"], n_rows, p=[0.95, 0.05]),
            "balance": rng.integers(-1000, 20000, n_rows),
            "housing": rng.choice(["no", "yes"], n_rows),
            "loan": rng.choice(["no", "yes"], n_rows, p=[0.8, 0.2]),
            "contact": rng.choice(["cellular", "telephone", "unknown"], n_rows),
            "day": rng.integers(1, 32, n_rows),
            "month": rng.choice(["may", "jun", "jul", "aug"], n_rows),
            "duration": rng.integers(0, 1500, n_rows),
            "campaign": rng.integers(1, 10, n_rows),
        }
    )
    signal = 1.5 * (df["age"] > 60) + (df["housing"] == "no") + df["duration"] / 400
    score = signal + rng.normal(0, 1, n_rows)
    df["y"] = np.where(score > np.quantile(score, 0.85), "yes", "no")
    return df


@pytest.fixture
def calls_df() -> pd.DataFrame:
    return make_calls()


@pytest.fixture
def calls_csv(calls_df, tmp_path):
    path = tmp_path / "calls.csv"
    calls_df.to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def trained_dir(tmp_path_factory):
    """Both models trained once on the synthetic data, shared by the train/predict tests."""
    folder = tmp_path_factory.mktemp("trained")
    make_calls().to_csv(folder / "calls.csv", index=False)
    train(folder / "calls.csv", folder / "models")
    return folder