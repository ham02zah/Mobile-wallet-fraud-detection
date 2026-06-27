import os
from dotenv import load_dotenv


load_dotenv()


DATA_PATH = "data/paysim.csv"
MODEL_PATH = "models/fraud_model.joblib"

REPORTS_DIR = "reports"
VISUALIZATIONS_DIR = "visualizations"

RANDOM_STATE = 42

TARGET_COLUMN = "isFraud"

MAX_TRAINING_ROWS = int(os.getenv("MAX_TRAINING_ROWS", "300000"))
FRAUD_MODEL_THRESHOLD = float(os.getenv("FRAUD_MODEL_THRESHOLD", "0.50"))

NUMERIC_FEATURES = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "originBalanceChange",
    "destinationBalanceChange",
    "amountToOldBalanceRatio",
]

CATEGORICAL_FEATURES = [
    "type",
]