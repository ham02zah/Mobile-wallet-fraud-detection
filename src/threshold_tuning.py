import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score


def analyze_thresholds(y_true, y_prob, thresholds=None):
    """
    Analyze classification performance at multiple probability thresholds.

    Lower threshold:
    - catches more fraud
    - increases false positives

    Higher threshold:
    - fewer false positives
    - may miss more fraud
    """

    if thresholds is None:
        thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

    rows = []

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        rows.append(
            {
                "threshold": threshold,
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1_score": f1_score(y_true, y_pred, zero_division=0),
                "predicted_fraud_count": int(y_pred.sum()),
            }
        )

    return pd.DataFrame(rows)