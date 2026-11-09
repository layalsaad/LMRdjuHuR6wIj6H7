"""Score customers with a trained model.

Usage: term-deposit-predict INPUT [--stage {pre_call,post_call}] [--model PATH] [--output PATH]

Writes the input rows with three extra columns: `probability_subscribe` (the chance of buying),
`prediction` (1 = call, using the cut-off learned in training) and `priority_rank` (1 = call
first). If the input also has a complete `y` column, accuracy, precision and recall are reported.
Only load model files you created yourself: joblib files can execute code when loaded.
"""

import argparse
import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import BaseEstimator

from term_deposit.config import MODELS_DIR, STAGES, TARGET
from term_deposit.data import validate
from term_deposit.train import metadata_path, model_path

logger = logging.getLogger(__name__)


def load_model(path: Path) -> BaseEstimator:
    if not Path(path).exists():
        raise FileNotFoundError(f"Model not found: {path}. Run term-deposit-train first.")
    return joblib.load(path)


def predict(model: BaseEstimator, df: pd.DataFrame) -> pd.DataFrame:
    """Return `df` with `probability_subscribe`, `prediction` and `priority_rank` added.

    Only the columns the model was trained on must be present and valid; other columns (such
    as customer IDs, or an empty y column) are passed through unchanged.
    """
    features = list(model.feature_names_in_)
    customers = validate(df.drop(columns=TARGET, errors="ignore"), False, features)
    probability = model.predict_proba(customers)[:, list(model.classes_).index(1)]
    return df.assign(
        probability_subscribe=probability.round(4),
        prediction=model.predict(customers),
        priority_rank=pd.Series(probability, index=df.index)
        .rank(ascending=False, method="first")
        .astype(int),
    )


def unseen_values(df: pd.DataFrame, categories: dict[str, list]) -> dict[str, list]:
    """Category values in `df` that the model never saw in training, per column."""
    unseen = {col: sorted(set(df[col]) - set(known)) for col, known in categories.items()}
    return {col: values for col, values in unseen.items() if values}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score customers with a term-deposit model.")
    parser.add_argument("input", type=Path, help="CSV with customer data")
    parser.add_argument(
        "--stage", choices=list(STAGES), default="pre_call", help="which trained model to use"
    )
    parser.add_argument("--model", type=Path, help="model file (default: models/<stage>.joblib)")
    parser.add_argument(
        "--output", type=Path, default=Path("predictions.csv"), help="where to write predictions"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    path = args.model or model_path(MODELS_DIR, args.stage)

    try:
        df = pd.read_csv(args.input, encoding="utf-8-sig")
        model = load_model(path)
        scored = predict(model, df)
        labels = (
            validate(df, features=[])[TARGET]
            if TARGET in df.columns and df[TARGET].notna().all()
            else None
        )
    except (FileNotFoundError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")

    if metadata_path(path).exists():
        categories = json.loads(metadata_path(path).read_text())["categories"]
        for col, values in unseen_values(df, categories).items():
            logger.warning(
                "New values in %s, scored as 'none of the known values': %s", col, values
            )

    scored.to_csv(args.output, index=False)
    logger.info(
        "Wrote %d predictions to %s (%d marked to call)",
        len(scored),
        args.output,
        scored["prediction"].sum(),
    )
    if labels is not None:
        called = scored["prediction"] == 1
        logger.info(
            "On %d labelled rows: accuracy %.1f%%, precision %.1f%%, recall %.1f%%",
            len(labels),
            100 * (scored["prediction"] == labels).mean(),
            100 * labels[called].mean() if called.any() else 0.0,
            100 * called[labels == 1].mean(),
        )


if __name__ == "__main__":
    main()