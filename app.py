import os
import json

import joblib
import streamlit as st

from src.config import MODEL_PATH
from src.predict import predict_transaction


METRICS_PATH = "reports/metrics.json"

PAGES = [
    "Fraud Risk Prediction",
    "Threshold Analysis",
    "Model Insights",
    "Transaction Analytics",
    "Visualization Gallery",
]


st.set_page_config(
    page_title="Mobile Wallet Fraud AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(239, 68, 68, 0.30), transparent 28%),
        radial-gradient(circle at top right, rgba(37, 99, 235, 0.32), transparent 30%),
        radial-gradient(circle at bottom left, rgba(168, 85, 247, 0.25), transparent 32%),
        linear-gradient(135deg, #020617 0%, #111827 45%, #030712 100%);
    color: white;
}

header[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stDecoration"] {
    display: none;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

label,
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p,
.stSelectbox label,
.stTextInput label,
.stNumberInput label,
.stRadio label,
.stSlider label {
    color: #ffffff !important;
    font-weight: 700 !important;
}

.stTextInput input,
.stNumberInput input {
    background-color: #ffffff !important;
    color: #000000 !important;
    border-radius: 12px !important;
}

div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    color: #000000 !important;
    border-radius: 12px !important;
}

div[data-baseweb="select"] span {
    color: #000000 !important;
}

ul[role="listbox"] {
    background-color: #ffffff !important;
}

li[role="option"],
li[role="option"] span,
li[role="option"] div {
    color: #000000 !important;
    background-color: #ffffff !important;
}

li[role="option"]:hover,
li[role="option"]:hover span,
li[role="option"]:hover div {
    background-color: #dbeafe !important;
}

.main-title {
    font-size: 3rem;
    line-height: 1.05;
    font-weight: 800;
    margin-bottom: 0.6rem;
    background: linear-gradient(90deg, #ef4444, #60a5fa, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.main-subtitle {
    font-size: 1.05rem;
    color: #dbeafe;
    max-width: 980px;
    margin-bottom: 1.5rem;
}

.glass-card {
    padding: 1.5rem;
    border-radius: 24px;
    background: rgba(255, 255, 255, 0.075);
    border: 1px solid rgba(255, 255, 255, 0.14);
    box-shadow: 0 22px 45px rgba(0, 0, 0, 0.28);
    backdrop-filter: blur(16px);
    margin-bottom: 1.2rem;
}

.section-title {
    font-size: 1.55rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
    color: #ffffff;
}

.section-caption {
    color: #e5e7eb;
    font-size: 0.95rem;
}

.nav-title {
    font-size: 1.7rem;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 0.3rem;
}

.nav-subtitle {
    color: #dbeafe;
    font-size: 0.95rem;
    font-weight: 600;
    margin-bottom: 1rem;
}

.nav-section {
    color: #ffffff;
    font-size: 1.05rem;
    font-weight: 800;
    margin-top: 1rem;
    margin-bottom: 0.6rem;
}

.feature-box {
    margin-top: 1rem;
    padding: 1rem;
    border-radius: 18px;
    color: #e5e7eb;
    font-size: 0.95rem;
    line-height: 1.9;
    background: rgba(255, 255, 255, 0.07);
    border: 1px solid rgba(255, 255, 255, 0.12);
}

[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 28px !important;
    background:
        linear-gradient(180deg, rgba(30, 41, 59, 0.94), rgba(3, 7, 18, 0.90)),
        rgba(255, 255, 255, 0.08) !important;
    border: 1px solid rgba(255, 255, 255, 0.16) !important;
    box-shadow: 0 22px 50px rgba(0, 0, 0, 0.34) !important;
}

div[role="radiogroup"] label {
    background: rgba(255, 255, 255, 0.08) !important;
    border: 1px solid rgba(255, 255, 255, 0.14) !important;
    border-radius: 18px !important;
    padding: 0.9rem 1rem !important;
    margin-bottom: 0.75rem !important;
    width: 100% !important;
}

div[role="radiogroup"] label:hover {
    background: linear-gradient(90deg, #dc2626, #2563eb, #7c3aed) !important;
}

div[role="radiogroup"] p {
    color: #ffffff !important;
    font-weight: 800 !important;
}

.result-high {
    padding: 1.6rem;
    border-radius: 24px;
    background: linear-gradient(135deg, rgba(220, 38, 38, 0.35), rgba(127, 29, 29, 0.18));
    border: 1px solid rgba(248, 113, 113, 0.55);
}

.result-medium {
    padding: 1.6rem;
    border-radius: 24px;
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.30), rgba(239, 68, 68, 0.14));
    border: 1px solid rgba(251, 191, 36, 0.55);
}

.result-low {
    padding: 1.6rem;
    border-radius: 24px;
    background: linear-gradient(135deg, rgba(37, 99, 235, 0.25), rgba(34, 197, 94, 0.14));
    border: 1px solid rgba(96, 165, 250, 0.55);
}

.result-title {
    font-size: 1.7rem;
    font-weight: 800;
    margin-bottom: 0.4rem;
}

.result-text {
    font-size: 1rem;
    color: #e5e7eb;
}

.metric-card {
    padding: 1.2rem;
    border-radius: 20px;
    background:
        linear-gradient(135deg, rgba(239, 68, 68, 0.18), rgba(37, 99, 235, 0.13)),
        rgba(255, 255, 255, 0.07);
    border: 1px solid rgba(255, 255, 255, 0.14);
    text-align: center;
}

.metric-value {
    font-size: 1.8rem;
    font-weight: 800;
    color: #60a5fa;
}

.metric-label {
    font-size: 0.85rem;
    color: #e5e7eb;
}

div.stButton > button:first-child {
    width: 100%;
    border-radius: 18px;
    height: 3.1rem;
    border: none;
    font-size: 0.95rem;
    font-weight: 800;
    color: white;
    background: linear-gradient(90deg, #dc2626, #2563eb, #7c3aed);
}

.stTabs [data-baseweb="tab"] {
    border-radius: 999px;
    padding: 10px 18px;
    background: rgba(255, 255, 255, 0.08);
    color: white;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #dc2626, #2563eb);
}
</style>
"""


st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


if "menu_open" not in st.session_state:
    st.session_state.menu_open = True

if "page" not in st.session_state:
    st.session_state.page = "Fraud Risk Prediction"


def toggle_menu():
    st.session_state.menu_open = not st.session_state.menu_open


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def load_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH, "r") as file:
        return json.load(file)


def display_metric_card(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


model = load_model()
metrics = load_metrics()


top_col1, top_col2 = st.columns([0.16, 0.84])

with top_col1:
    button_label = "☰ Open Menu" if not st.session_state.menu_open else "✕ Close Menu"
    st.button(button_label, on_click=toggle_menu)

with top_col2:
    st.markdown(
        """
        <div class="main-title">Mobile Wallet Fraud Detection AI</div>
        <div class="main-subtitle">
            A FinTech risk scoring dashboard that predicts suspicious mobile wallet transactions,
            tunes fraud thresholds, and visualizes transaction fraud patterns.
        </div>
        """,
        unsafe_allow_html=True,
    )


if model is None:
    st.error("Model not found. Train first: `python3.11 -m src.train_model`")
    st.stop()


if st.session_state.menu_open:
    menu_col, content_col = st.columns([0.25, 0.75], gap="large")
else:
    content_col = st.container()


if st.session_state.menu_open:
    with menu_col:
        with st.container(border=True):
            st.markdown(
                """
                <div class="nav-title">🛡️ Fraud AI</div>
                <div class="nav-subtitle">Mobile Wallet Risk Dashboard</div>
                <div class="nav-section">Navigation</div>
                """,
                unsafe_allow_html=True,
            )

            selected_page = st.radio(
                label="Select dashboard section",
                options=PAGES,
                index=PAGES.index(st.session_state.page),
                label_visibility="collapsed",
            )

            st.session_state.page = selected_page

            st.markdown(
                """
                <div class="nav-section">Project Features</div>
                <div class="feature-box">
                    • Fraud probability scoring<br>
                    • Risk level classification<br>
                    • Threshold tuning<br>
                    • PR-AUC evaluation<br>
                    • Transaction analytics<br>
                    • Cybersecurity dashboard UI
                </div>
                """,
                unsafe_allow_html=True,
            )


with content_col:
    page = st.session_state.page

    if page == "Fraud Risk Prediction":
        left_col, right_col = st.columns([1.0, 1.0])

        with left_col:
            st.markdown(
                """
                <div class="glass-card">
                    <div class="section-title">Transaction Input</div>
                    <div class="section-caption">
                        Enter transaction details to calculate fraud probability and recommended action.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            transaction_type = st.selectbox("Transaction Type", ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"])
            step = st.number_input("Transaction Step / Hour", min_value=1, value=1)
            amount = st.number_input("Transaction Amount", min_value=0.0, value=5000.0)
            oldbalance_org = st.number_input("Sender Old Balance", min_value=0.0, value=10000.0)
            newbalance_orig = st.number_input("Sender New Balance", min_value=0.0, value=5000.0)
            oldbalance_dest = st.number_input("Receiver Old Balance", min_value=0.0, value=0.0)
            newbalance_dest = st.number_input("Receiver New Balance", min_value=0.0, value=5000.0)

            threshold = st.slider("Fraud Decision Threshold", 0.05, 0.95, 0.50, 0.05)

            predict_button = st.button("Score Transaction Risk")

        with right_col:
            st.markdown(
                """
                <div class="glass-card">
                    <div class="section-title">Risk Scoring Output</div>
                    <div class="section-caption">
                        Fraud probability, risk level, and recommended business action.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if predict_button:
                transaction_data = {
                    "step": step,
                    "type": transaction_type,
                    "amount": amount,
                    "oldbalanceOrg": oldbalance_org,
                    "newbalanceOrig": newbalance_orig,
                    "oldbalanceDest": oldbalance_dest,
                    "newbalanceDest": newbalance_dest,
                }

                result = predict_transaction(transaction_data, threshold=threshold)

                fraud_probability = result["fraud_probability"]
                non_fraud_probability = result["non_fraud_probability"]

                if result["risk_level"] == "High Risk":
                    card_class = "result-high"
                    icon = "🚨"
                elif result["risk_level"] == "Medium Risk":
                    card_class = "result-medium"
                    icon = "⚠️"
                else:
                    card_class = "result-low"
                    icon = "✅"

                st.markdown(
                    f"""
                    <div class="{card_class}">
                        <div class="result-title">{icon} {result["risk_level"]}</div>
                        <div class="result-text">Fraud Probability: {fraud_probability:.2%}</div>
                        <div class="result-text">Suggested Action: {result["suggested_action"]}</div>
                        <div class="result-text">Threshold Used: {result["threshold_used"]:.2f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.write("")
                st.progress(float(fraud_probability))

                c1, c2 = st.columns(2)

                with c1:
                    display_metric_card("Fraud Score", f"{fraud_probability:.1%}")
                with c2:
                    display_metric_card("Non-Fraud Score", f"{non_fraud_probability:.1%}")

            else:
                st.info("Enter transaction details and score risk.")

    elif page == "Threshold Analysis":
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title">Threshold Analysis</div>
                <div class="section-caption">
                    Understand precision, recall, and F1-score tradeoffs at different fraud thresholds.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        path = "visualizations/threshold_analysis.png"

        if os.path.exists(path):
            st.image(path, use_container_width=True)
        else:
            st.warning("Threshold analysis graph not found. Train model first.")

        csv_path = "reports/threshold_analysis.csv"

        if os.path.exists(csv_path):
            st.dataframe(__import__("pandas").read_csv(csv_path), use_container_width=True)

    elif page == "Model Insights":
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title">Model Performance Summary</div>
                <div class="section-caption">
                    Fraud detection metrics focused on imbalanced classification.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if metrics:
            st.write(f"### Best Model: `{metrics['best_model']}`")

            c1, c2, c3, c4, c5 = st.columns(5)

            with c1:
                display_metric_card("Precision", f"{metrics['precision']:.3f}")
            with c2:
                display_metric_card("Recall", f"{metrics['recall']:.3f}")
            with c3:
                display_metric_card("F1 Score", f"{metrics['f1_score']:.3f}")
            with c4:
                display_metric_card("ROC-AUC", f"{metrics['roc_auc']:.3f}")
            with c5:
                display_metric_card("PR-AUC", f"{metrics['pr_auc']:.3f}")
        else:
            st.warning("Metrics not found. Train model first.")

        insight_graphs = {
            "Model Comparison": "visualizations/model_comparison.png",
            "Feature Importance": "visualizations/feature_importance.png",
            "Confusion Matrix": "visualizations/confusion_matrix.png",
            "ROC Curve": "visualizations/roc_curve.png",
            "Precision-Recall Curve": "visualizations/precision_recall_curve.png",
        }

        for title, path in insight_graphs.items():
            if os.path.exists(path):
                st.markdown(f"### {title}")
                st.image(path, use_container_width=True)

    elif page == "Transaction Analytics":
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title">Transaction Analytics</div>
                <div class="section-caption">
                    Explore fraud behavior across transaction amount and transaction type.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        graphs = {
            "Fraud Distribution": "visualizations/fraud_distribution.png",
            "Transaction Type Distribution": "visualizations/transaction_type_distribution.png",
            "Fraud by Transaction Type": "visualizations/fraud_by_transaction_type.png",
            "Amount Distribution": "visualizations/amount_distribution.png",
        }

        for title, path in graphs.items():
            if os.path.exists(path):
                st.markdown(f"### {title}")
                st.image(path, use_container_width=True)

    elif page == "Visualization Gallery":
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title">Visualization Gallery</div>
                <div class="section-caption">
                    View all fraud analytics and model evaluation visualizations.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        graph_paths = {
            "Fraud Distribution": "visualizations/fraud_distribution.png",
            "Transaction Type Distribution": "visualizations/transaction_type_distribution.png",
            "Fraud by Transaction Type": "visualizations/fraud_by_transaction_type.png",
            "Amount Distribution": "visualizations/amount_distribution.png",
            "Confusion Matrix": "visualizations/confusion_matrix.png",
            "ROC Curve": "visualizations/roc_curve.png",
            "Precision-Recall Curve": "visualizations/precision_recall_curve.png",
            "Threshold Analysis": "visualizations/threshold_analysis.png",
            "Model Comparison": "visualizations/model_comparison.png",
            "Feature Importance": "visualizations/feature_importance.png",
        }

        tabs = st.tabs(list(graph_paths.keys()))

        for tab, (title, path) in zip(tabs, graph_paths.items()):
            with tab:
                if os.path.exists(path):
                    st.image(path, use_container_width=True)
                else:
                    st.warning(f"{title} graph not found. Train model first.")