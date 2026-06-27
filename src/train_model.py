import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    DATA_PATH,
    MODEL_PATH,
    REPORTS_DIR,
    VISUALIZATIONS_DIR,
    RANDOM_STATE,
    MAX_TRAINING_ROWS,
    TARGET_COLUMN,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)
from src.data_processing import (
    load_fraud_dataset,
    clean_fraud_data,
    create_training_sample,
    prepare_features_and_target,
)
from src.threshold_tuning import analyze_thresholds


def create_directories():
    os.makedirs("models", exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)


def plot_fraud_distribution(df):
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x=TARGET_COLUMN, hue=TARGET_COLUMN, palette="rocket", legend=False)
    plt.title("Fraud vs Non-Fraud Distribution")
    plt.xlabel("0 = Non-Fraud, 1 = Fraud")
    plt.ylabel("Transactions")
    plt.tight_layout()
    plt.savefig(f"{VISUALIZATIONS_DIR}/fraud_distribution.png")
    plt.close()


def plot_transaction_type_distribution(df):
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="type", hue="type", palette="mako", legend=False)
    plt.title("Transaction Type Distribution")
    plt.xlabel("Transaction Type")
    plt.ylabel("Transactions")
    plt.tight_layout()
    plt.savefig(f"{VISUALIZATIONS_DIR}/transaction_type_distribution.png")
    plt.close()


def plot_fraud_by_transaction_type(df):
    fraud_rate = (
        df.groupby("type")[TARGET_COLUMN]
        .mean()
        .reset_index()
        .sort_values(by=TARGET_COLUMN, ascending=False)
    )

    plt.figure(figsize=(8, 5))
    sns.barplot(data=fraud_rate, x="type", y=TARGET_COLUMN, hue="type", palette="viridis", legend=False)
    plt.title("Fraud Rate by Transaction Type")
    plt.xlabel("Transaction Type")
    plt.ylabel("Fraud Rate")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f"{VISUALIZATIONS_DIR}/fraud_by_transaction_type.png")
    plt.close()


def plot_amount_distribution(df):
    plot_df = df[df["amount"] <= df["amount"].quantile(0.99)]

    plt.figure(figsize=(8, 5))
    sns.histplot(data=plot_df, x="amount", hue=TARGET_COLUMN, bins=50, kde=True, palette="rocket")
    plt.title("Transaction Amount Distribution")
    plt.xlabel("Amount")
    plt.ylabel("Transactions")
    plt.tight_layout()
    plt.savefig(f"{VISUALIZATIONS_DIR}/amount_distribution.png")
    plt.close()


def build_pipeline(model):
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "pr_auc": float(average_precision_score(y_test, y_prob)),
    }

    return metrics, y_pred, y_prob


def plot_confusion_matrix_graph(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Non-Fraud", "Fraud"],
        yticklabels=["Non-Fraud", "Fraud"],
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{VISUALIZATIONS_DIR}/confusion_matrix.png")
    plt.close()


def plot_roc_curve_graph(y_test, y_prob):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc_score:.4f}", color="red")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{VISUALIZATIONS_DIR}/roc_curve.png")
    plt.close()


def plot_precision_recall_curve_graph(y_test, y_prob):
    precision, recall, _ = precision_recall_curve(y_test, y_prob)

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, color="purple")
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig(f"{VISUALIZATIONS_DIR}/precision_recall_curve.png")
    plt.close()


def plot_model_comparison(comparison_df):
    metric_columns = ["precision", "recall", "f1_score", "roc_auc", "pr_auc"]

    plot_df = comparison_df.melt(
        id_vars="model",
        value_vars=metric_columns,
        var_name="metric",
        value_name="score",
    )

    plt.figure(figsize=(11, 6))
    sns.barplot(data=plot_df, x="model", y="score", hue="metric", palette="viridis")
    plt.title("Fraud Detection Model Comparison")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(f"{VISUALIZATIONS_DIR}/model_comparison.png")
    plt.close()


def plot_threshold_analysis(threshold_df):
    plt.figure(figsize=(9, 5))

    plt.plot(threshold_df["threshold"], threshold_df["precision"], marker="o", label="Precision")
    plt.plot(threshold_df["threshold"], threshold_df["recall"], marker="o", label="Recall")
    plt.plot(threshold_df["threshold"], threshold_df["f1_score"], marker="o", label="F1 Score")

    plt.title("Threshold Analysis")
    plt.xlabel("Fraud Probability Threshold")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{VISUALIZATIONS_DIR}/threshold_analysis.png")
    plt.close()


def plot_feature_importance(best_model):
    model = best_model.named_steps["model"]

    if not hasattr(model, "feature_importances_"):
        return

    preprocessor = best_model.named_steps["preprocessor"]
    encoder = preprocessor.named_transformers_["categorical"].named_steps["encoder"]
    categorical_names = encoder.get_feature_names_out(CATEGORICAL_FEATURES)

    feature_names = NUMERIC_FEATURES + list(categorical_names)

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values(by="importance", ascending=False).head(15)

    plt.figure(figsize=(9, 7))
    sns.barplot(data=importance_df, x="importance", y="feature", hue="feature", palette="rocket", legend=False)
    plt.title("Top Fraud Detection Features")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(f"{VISUALIZATIONS_DIR}/feature_importance.png")
    plt.close()


def train_model():
    create_directories()

    print("Loading dataset...")
    raw_df = load_fraud_dataset(DATA_PATH)

    print("Cleaning dataset...")
    cleaned_df = clean_fraud_data(raw_df)

    print("Creating training sample...")
    df = create_training_sample(cleaned_df, MAX_TRAINING_ROWS, RANDOM_STATE)

    print("Creating exploratory visualizations...")
    plot_fraud_distribution(df)
    plot_transaction_type_distribution(df)
    plot_fraud_by_transaction_type(df)
    plot_amount_distribution(df)

    X, y = prepare_features_and_target(
        df,
        NUMERIC_FEATURES,
        CATEGORICAL_FEATURES,
        TARGET_COLUMN,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    candidate_models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
    }

    comparison_results = []
    trained_models = {}

    for model_name, estimator in candidate_models.items():
        print(f"Training {model_name}...")

        pipeline = build_pipeline(estimator)
        pipeline.fit(X_train, y_train)

        metrics, _, _ = evaluate_model(pipeline, X_test, y_test)

        comparison_results.append(
            {
                "model": model_name,
                **metrics,
            }
        )

        trained_models[model_name] = pipeline

    comparison_df = pd.DataFrame(comparison_results)
    comparison_df.to_csv(f"{REPORTS_DIR}/model_comparison.csv", index=False)

    best_model_name = comparison_df.sort_values(
        by="pr_auc",
        ascending=False,
    ).iloc[0]["model"]

    best_model = trained_models[best_model_name]

    best_metrics, y_pred, y_prob = evaluate_model(best_model, X_test, y_test)

    print("\nBest model:", best_model_name)
    print(classification_report(y_test, y_pred, target_names=["Non-Fraud", "Fraud"]))

    threshold_df = analyze_thresholds(y_test, y_prob)
    threshold_df.to_csv(f"{REPORTS_DIR}/threshold_analysis.csv", index=False)

    final_report = {
        "best_model": best_model_name,
        "training_rows_used": int(len(df)),
        **best_metrics,
    }

    with open(f"{REPORTS_DIR}/metrics.json", "w") as file:
        json.dump(final_report, file, indent=4)

    plot_confusion_matrix_graph(y_test, y_pred)
    plot_roc_curve_graph(y_test, y_prob)
    plot_precision_recall_curve_graph(y_test, y_prob)
    plot_model_comparison(comparison_df)
    plot_threshold_analysis(threshold_df)
    plot_feature_importance(best_model)

    joblib.dump(best_model, MODEL_PATH)

    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_model()