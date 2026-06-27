import joblib
import pandas as pd

from src.config import MODEL_PATH, FRAUD_MODEL_THRESHOLD


def load_model():
    """Load trained fraud model."""
    return joblib.load(MODEL_PATH)


def get_risk_level(fraud_probability):
    """Convert fraud probability into risk level and action."""
    if fraud_probability >= 0.75:
        return "High Risk", "Block transaction or send to manual fraud review"

    if fraud_probability >= 0.40:
        return "Medium Risk", "Hold transaction and request additional verification"

    return "Low Risk", "Approve transaction"


def engineer_single_transaction(transaction_data):
    """Create engineered features for one transaction."""
    data = transaction_data.copy()

    data["originBalanceChange"] = data["oldbalanceOrg"] - data["newbalanceOrig"]
    data["destinationBalanceChange"] = data["newbalanceDest"] - data["oldbalanceDest"]

    if data["oldbalanceOrg"] == 0:
        data["amountToOldBalanceRatio"] = 0
    else:
        data["amountToOldBalanceRatio"] = data["amount"] / data["oldbalanceOrg"]

    return data


def predict_transaction(transaction_data, threshold=FRAUD_MODEL_THRESHOLD):
    """Predict fraud risk for a transaction."""
    model = load_model()

    engineered_data = engineer_single_transaction(transaction_data)

    input_df = pd.DataFrame([engineered_data])

    probability = model.predict_proba(input_df)[0]

    non_fraud_probability = probability[0]
    fraud_probability = probability[1]

    prediction = int(fraud_probability >= threshold)

    risk_level, suggested_action = get_risk_level(fraud_probability)

    return {
        "prediction": prediction,
        "fraud_probability": fraud_probability,
        "non_fraud_probability": non_fraud_probability,
        "risk_level": risk_level,
        "suggested_action": suggested_action,
        "threshold_used": threshold,
    }