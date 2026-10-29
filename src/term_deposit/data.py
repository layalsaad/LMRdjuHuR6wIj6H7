"""Loading and validating customer call data."""

from pathlib import Path

import pandas as pd

from term_deposit.config import BINARY, DATA_PATH, FEATURES, NUMERIC, TARGET, VALUE_RANGES, YES_NO


def load_data(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Read the labelled call data and return it validated, with y as 1 (yes) or 0 (no)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}. "
            "Place term-deposit-marketing-2020.csv there (see README)."
        )
    # utf-8-sig transparently strips a byte-order mark if the export added one.
    return validate(pd.read_csv(path, encoding="utf-8-sig"))


def validate(
    df: pd.DataFrame, require_target: bool = True, features: list[str] = FEATURES
) -> pd.DataFrame:
    """Check the columns and values of call data.

    `features` lists the columns that must be present: the pre-call model does not need
    `duration`. Returns a copy restricted to those columns, plus the target when present,
    recoded as 1 (yes) and 0 (no). Raises ValueError listing every problem found.
    """
    features = list(features)
    required = [*features, TARGET] if require_target else features
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    columns = [*features, TARGET] if TARGET in df.columns else features
    df = df[columns].copy()

    errors = []
    with_nulls = [col for col in columns if df[col].isna().any()]
    if with_nulls:
        errors.append(f"missing values in {with_nulls}")

    complete = [col for col in columns if col not in with_nulls]
    numeric = [col for col in complete if col in NUMERIC]
    non_integer = [col for col in numeric if not pd.api.types.is_integer_dtype(df[col])]
    if non_integer:
        errors.append(f"non-integer values in {non_integer}")

    # Range checks only make sense on complete integer columns.
    clean = [col for col in numeric if col not in non_integer]
    out_of_range = [
        col
        for col, (low, high) in VALUE_RANGES.items()
        if col in clean and not df[col].between(low, high).all()
    ]
    if out_of_range:
        allowed = ", ".join(f"{col} {VALUE_RANGES[col]}" for col in out_of_range)
        errors.append(f"values outside the allowed range in {allowed}")

    yes_no = [col for col in complete if col in BINARY or col == TARGET]
    not_yes_no = [col for col in yes_no if not df[col].isin(YES_NO).all()]
    if not_yes_no:
        errors.append(f"values other than 'yes'/'no' in {not_yes_no}")

    if errors:
        raise ValueError("Invalid call data: " + "; ".join(errors))
    if TARGET in df.columns:
        df[TARGET] = (df[TARGET] == "yes").astype(int)
    return df