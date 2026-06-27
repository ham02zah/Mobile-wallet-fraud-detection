import numpy as np
import pandas as pd


def load_fraud_dataset(data_path):
    """Load PaySim mobile money fraud dataset."""
    df = pd.read_csv(data_path)

    required_columns = [
        "step",
        "type",
        "amount",
        "nameOrig",
        "oldbalanceOrg",
        "newbalanceOrig",
        "nameDest",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud",
        "isFlaggedFraud",
    ]

    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return df


def clean_fraud_data(df):
    """
    Clean and engineer transaction features.

    Feature notes:
    - originBalanceChange shows balance movement from sender account
    - destinationBalanceChange shows balance movement in receiver account
    - amountToOldBalanceRatio helps detect suspicious transfers
    """

    data = df.copy()

    numeric_columns = [
        "step",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=numeric_columns + ["type"])

    data["originBalanceChange"] = data["oldbalanceOrg"] - data["newbalanceOrig"]
    data["destinationBalanceChange"] = data["newbalanceDest"] - data["oldbalanceDest"]

    data["amountToOldBalanceRatio"] = data["amount"] / data["oldbalanceOrg"].replace(0, np.nan)
    data["amountToOldBalanceRatio"] = data["amountToOldBalanceRatio"].replace(
        [np.inf, -np.inf],
        np.nan,
    )
    data["amountToOldBalanceRatio"] = data["amountToOldBalanceRatio"].fillna(0)

    data["type"] = data["type"].astype(str)

    return data


def create_training_sample(df, max_rows, random_state):
    """
    Create a laptop-friendly training sample.

    PaySim has millions of rows. For VS Code/MacBook training, we keep:
    - all fraud rows
    - a random sample of non-fraud rows
    """

    if len(df) <= max_rows:
        return df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    fraud_df = df[df["isFraud"] == 1]
    non_fraud_df = df[df["isFraud"] == 0]

    remaining_rows = max_rows - len(fraud_df)

    if remaining_rows <= 0:
        sampled_df = fraud_df.sample(n=max_rows, random_state=random_state)
    else:
        sampled_non_fraud = non_fraud_df.sample(
            n=min(remaining_rows, len(non_fraud_df)),
            random_state=random_state,
        )
        sampled_df = pd.concat([fraud_df, sampled_non_fraud], axis=0)

    sampled_df = sampled_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return sampled_df


def prepare_features_and_target(df, numeric_features, categorical_features, target_column):
    """Split data into features and target."""
    feature_columns = numeric_features + categorical_features

    X = df[feature_columns]
    y = df[target_column].astype(int)

    return X, y