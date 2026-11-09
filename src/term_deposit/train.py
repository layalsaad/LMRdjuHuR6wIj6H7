"""Train the production models on all labelled customers and save them with their metadata.

Usage: term-deposit-train [--data PATH] [--output-dir DIR] [--stage {pre_call,post_call,all}]

Run it again after every campaign, on a file that includes the new results, so the models
learn the current timing (notebook 02 shows that performance drifts between periods).
"""

import argparse
import hashlib
import json
import logging
import platform
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.model_selection import TunedThresholdClassifierCV

from term_deposit.config import CATEGORICAL, DATA_PATH, FINAL_MODEL, MODELS_DIR, STAGES, TARGET
from term_deposit.data import load_data
from term_deposit.evaluation import SCORING, cv_scores
from term_deposit.models import get_candidates, with_threshold

logger = logging.getLogger(__name__)


def build_model(stage: str) -> TunedThresholdClassifierCV:
    """The model chosen in notebook 02 for `stage`, tuning its settings and its cut-off."""
    features = STAGES[stage]
    return with_threshold(get_candidates()[FINAL_MODEL[stage]].tuned(features))


def model_path(output_dir: Path, stage: str) -> Path:
    return Path(output_dir) / f"{stage}.joblib"


def metadata_path(path: Path) -> Path:
    return Path(path).with_suffix(".json")


def train_stage(df: pd.DataFrame, stage: str, data_sha256: str, output_dir: Path) -> dict:
    """Score the full procedure with 5-fold CV, then fit it on every customer and save it."""
    features = STAGES[stage]
    X, y = df[features], df[TARGET]
    model = build_model(stage)

    scores = cv_scores(model, X, y)
    model.fit(X, y)

    path = model_path(output_dir, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    metadata = {
        "stage": stage,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "features": features,
        "model": FINAL_MODEL[stage],
        "settings": {
            name.removeprefix("model__"): v for name, v in model.estimator_.best_params_.items()
        },
        "threshold": round(float(model.best_threshold_), 4),
        "training_rows": len(df),
        "subscribe_share": round(float(y.mean()), 4),
        "data_sha256": data_sha256,
        # Values seen in training, so predict can warn about new ones (e.g. a new job title).
        "categories": {col: sorted(df[col].unique().tolist()) for col in CATEGORICAL if col in X},
        "cross_validation": {
            "folds": len(scores),
            **{
                metric: {
                    "mean": round(float(scores[metric].mean()), 4),
                    "std": round(float(scores[metric].std()), 4),
                }
                for metric in SCORING
            },
        },
        "versions": {
            "term-deposit": version("term-deposit"),
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
        },
    }
    metadata_path(path).write_text(json.dumps(metadata, indent=2) + "\n")
    # Logged from the unrounded scores, so the numbers match notebook 02 exactly.
    logger.info(
        "%s: trained on %d customers. 5-fold accuracy %.1f%%, precision %.1f%%, recall %.1f%%, "
        "PR-AUC %.3f. Cut-off %.2f. Saved %s",
        stage,
        len(df),
        100 * scores["accuracy"].mean(),
        100 * scores["precision"].mean(),
        100 * scores["recall"].mean(),
        scores["average_precision"].mean(),
        model.best_threshold_,
        path,
    )
    return metadata


def train(
    data_path: Path = DATA_PATH,
    output_dir: Path = MODELS_DIR,
    stages: tuple[str, ...] = tuple(STAGES),
) -> dict[str, dict]:
    """Train and save one model per stage; returns each stage's metadata."""
    df = load_data(data_path)
    data_sha256 = hashlib.sha256(Path(data_path).read_bytes()).hexdigest()
    return {stage: train_stage(df, stage, data_sha256, output_dir) for stage in stages}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the term-deposit models.")
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="labelled customer CSV")
    parser.add_argument(
        "--output-dir", type=Path, default=MODELS_DIR, help="folder for the model files"
    )
    parser.add_argument(
        "--stage", choices=[*STAGES, "all"], default="all", help="which model to train"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    stages = tuple(STAGES) if args.stage == "all" else (args.stage,)

    try:
        train(args.data, args.output_dir, stages)
    except (FileNotFoundError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()