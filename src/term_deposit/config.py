"""Project-wide constants: file locations, the data schema and evaluation settings."""

import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "term-deposit-marketing-2020.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

TARGET = "y"
YES_NO = ("no", "yes")

# Column groups decide how each column is checked and, later, encoded for the models.
NUMERIC = ["age", "balance", "day", "duration", "campaign"]
BINARY = ["default", "housing", "loan"]
CATEGORICAL = ["job", "marital", "education", "contact", "month"]
FEATURES = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "contact",
    "day",
    "month",
    "duration",
    "campaign",
]

# Smallest and largest valid value. A value outside these is a data error, not an unusual
# customer. balance has no limits: it is negative when the account is overdrawn.
VALUE_RANGES = {
    "age": (18, 100),
    "day": (1, 31),
    "duration": (0, math.inf),
    "campaign": (1, math.inf),
}

# duration is only known once the call has ended, so it cannot help decide whom to call.
# pre_call ranks customers before dialling; post_call decides whom to call back.
CALL_RESULT_FEATURES = ["duration"]
STAGES = {
    "pre_call": [col for col in FEATURES if col not in CALL_RESULT_FEATURES],
    "post_call": FEATURES,
}

# Evaluation: the brief asks for the average over 5-fold cross-validation.
RANDOM_STATE = 42
CV_SPLITS = 5