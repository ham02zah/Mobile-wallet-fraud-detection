# Mobile Wallet Fraud Detection and Transaction Risk Scoring Dashboard

## Live app

Link: https://mobile-wallet-fraud-detection-age8kgvjke5rwuhbx9hpzs.streamlit.app

## Project Overview

This project is an intermediate-level machine learning dashboard that detects suspicious mobile wallet transactions and assigns transaction risk levels.

The system is inspired by mobile wallet and telecom-fintech platforms such as JazzCash, Easypaisa, and other digital payment services. It predicts whether a transaction is likely fraudulent, calculates fraud probability, classifies risk level, and suggests a business action such as approve, review, or block.

The project includes data preprocessing, transaction-level feature engineering, model comparison, imbalanced classification evaluation, threshold tuning, fraud analytics visualizations, a Streamlit dashboard, Docker setup, and Kubernetes deployment files.

---

## Dataset

Dataset used:

PaySim Mobile Money Fraud Dataset  
https://www.kaggle.com/datasets/ealaxi/paysim1

Download the dataset and place it here:

```text
data/paysim.csv
```

If your downloaded file is named:

```text
PS_20174392719_1491204439457_log.csv
```

rename it to:

```text
paysim.csv
```

---

## Dataset Columns

The dataset contains mobile money transaction fields such as:

- `step`
- `type`
- `amount`
- `nameOrig`
- `oldbalanceOrg`
- `newbalanceOrig`
- `nameDest`
- `oldbalanceDest`
- `newbalanceDest`
- `isFraud`
- `isFlaggedFraud`

Target column:

```text
isFraud
```

Where:

```text
0 = Non-Fraud
1 = Fraud
```

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib
- Python-dotenv
- Docker
- Kubernetes

---

## Machine Learning Workflow

1. Load PaySim mobile money transaction dataset
2. Clean transaction records
3. Engineer transaction risk features
4. Handle class imbalance using model class weights
5. Train multiple fraud detection models
6. Compare models using fraud-focused metrics
7. Select best model based on PR-AUC
8. Perform threshold tuning
9. Save trained model
10. Generate fraud analytics visualizations
11. Build Streamlit risk scoring dashboard
12. Add Docker and Kubernetes deployment files

---

## Feature Engineering

The project creates additional transaction-level features:

### `originBalanceChange`

```text
oldbalanceOrg - newbalanceOrig
```

This shows how much money moved out of the sender account.

### `destinationBalanceChange`

```text
newbalanceDest - oldbalanceDest
```

This shows how much money moved into the receiver account.

### `amountToOldBalanceRatio`

```text
amount / oldbalanceOrg
```

This helps detect suspicious transactions where the amount is unusually large compared to sender balance.

---

## Models Compared

The project compares:

- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier

The best model is selected based on:

```text
PR-AUC
```

PR-AUC is preferred because fraud detection is a highly imbalanced classification problem.

---

## Evaluation Metrics

Accuracy alone is not reliable for fraud detection because most transactions are non-fraud.

This project evaluates models using:

- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion Matrix
- Precision-Recall Curve
- Threshold Analysis

---

## Threshold Tuning

The dashboard supports fraud probability threshold analysis.

Lower threshold:

- catches more fraud
- increases false positives

Higher threshold:

- reduces false positives
- may miss more fraud

The project saves threshold analysis here:

```text
reports/threshold_analysis.csv
```

And visualization here:

```text
visualizations/threshold_analysis.png
```

---

## Risk Scoring Logic

The app converts fraud probability into a risk level:

```text
Low Risk
Medium Risk
High Risk
```

Suggested actions:

```text
Low Risk    → Approve transaction
Medium Risk → Hold transaction and request additional verification
High Risk   → Block transaction or send to manual fraud review
```

---

## Dashboard Features

The Streamlit dashboard includes:

- Fraud risk prediction
- Fraud probability score
- Non-fraud probability score
- Risk level classification
- Suggested business action
- Adjustable fraud decision threshold
- Threshold analysis page
- Model insights page
- Transaction analytics page
- Visualization gallery
- Cybersecurity-style multi-gradient UI

---

## Project Structure

```text
mobile-wallet-fraud-detection/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
├── Dockerfile
│
├── data/
│   └── paysim.csv
│
├── models/
│   └── fraud_model.joblib
│
├── reports/
│   ├── metrics.json
│   ├── model_comparison.csv
│   └── threshold_analysis.csv
│
├── visualizations/
│   ├── fraud_distribution.png
│   ├── transaction_type_distribution.png
│   ├── fraud_by_transaction_type.png
│   ├── amount_distribution.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   ├── threshold_analysis.png
│   ├── model_comparison.png
│   └── feature_importance.png
│
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
│
└── src/
    ├── __init__.py
    ├── config.py
    ├── data_processing.py
    ├── threshold_tuning.py
    ├── train_model.py
    └── predict.py
```

---

## Environment Variables

Create `.env` from `.env.example`:

```bash
cp .env.example .env
```

Example `.env`:

```env
FRAUD_MODEL_THRESHOLD=0.50
MAX_TRAINING_ROWS=300000
```

### `FRAUD_MODEL_THRESHOLD`

Default probability threshold used for fraud prediction.

### `MAX_TRAINING_ROWS`

Limits training rows so the project can run on a normal laptop.

---

## How to Run Locally in VS Code

### 1. Install dependencies

```bash
python3.11 -m pip install -r requirements.txt
```

### 2. Add dataset

Download the PaySim dataset from Kaggle:

https://www.kaggle.com/datasets/ealaxi/paysim1

Place it here:

```text
data/paysim.csv
```

### 3. Train the model

```bash
python3.11 -m src.train_model
```

This creates:

```text
models/fraud_model.joblib
reports/metrics.json
reports/model_comparison.csv
reports/threshold_analysis.csv
visualizations/
```

### 4. Run the Streamlit app

```bash
python3.11 -m streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

## Docker Setup

Docker packages the project so it can run in a consistent environment.

### 1. Build Docker image

Make sure the model is trained first:

```bash
python3.11 -m src.train_model
```

Then build:

```bash
docker build -t mobile-wallet-fraud-ai:latest .
```

### 2. Run Docker container

```bash
docker run -p 8501:8501 mobile-wallet-fraud-ai:latest
```

Open:

```text
http://localhost:8501
```

---

## Kubernetes Setup

Kubernetes files are stored in:

```text
k8s/
```

Before applying Kubernetes files, make sure Kubernetes is running.

For Docker Desktop:

1. Open Docker Desktop
2. Go to Settings
3. Enable Kubernetes
4. Apply and restart

Check cluster:

```bash
kubectl cluster-info
kubectl get nodes
```

### Apply deployment and service

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

### Check pods

```bash
kubectl get pods
```

### Port forward service

```bash
kubectl port-forward service/mobile-wallet-fraud-service 8501:80
```

Open:

```text
http://localhost:8501
```

---

## Common Kubernetes Issue

If you see:

```text
connect: connection refused localhost:8080
```

It means Kubernetes is not running or kubectl is not connected to a cluster.

Fix:

```bash
kubectl config current-context
kubectl cluster-info
```

If using Docker Desktop:

```bash
kubectl config use-context docker-desktop
```

---

## Example Transaction Input

```text
Transaction Type: TRANSFER
Step: 1
Amount: 5000
Sender Old Balance: 10000
Sender New Balance: 5000
Receiver Old Balance: 0
Receiver New Balance: 5000
Threshold: 0.50
```

Example output:

```text
Risk Level: Low Risk
Fraud Probability: 8.4%
Suggested Action: Approve transaction
```

---

## Business Use Case

A mobile wallet company can use this type of system to:

- detect suspicious transactions
- reduce financial fraud
- prioritize manual review
- tune fraud thresholds
- reduce false positives
- protect customers from account misuse

---

## Future Improvements

- Add XGBoost or LightGBM
- Add anomaly detection model
- Add graph-based fraud detection
- Add real-time FastAPI endpoint
- Add database logging
- Add user authentication
- Add cloud deployment
- Add model drift monitoring
- Add transaction batch upload
- Add fraud investigation dashboard

---

## Author

HAMZAH JAWAD
