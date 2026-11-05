
from pathlib import Path

import pandas as pd
import json
import requests
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]


API_URL = "http://localhost:8000/monitoring/predictions?limit=1000"
METRICS_API_URL = "http://localhost:8000/monitoring/metrics"
DRIFT_REPORT_PATH = PROJECT_ROOT / "reports" / "data_drift_report.html"
DRIFT_ALERT_PATH = PROJECT_ROOT / "reports" / "drift_alerts.json"
TEST_EVALUATION_PATH = PROJECT_ROOT / "reports" / "test_evaluation.json"

st.set_page_config(
    page_title="Olist ML Monitoring",
    page_icon="📦",
    layout="wide"
)

st.title("📦 Olist Delivery Risk — Monitoring Dashboard")
st.caption("Live prediction monitoring through FastAPI.")

st.divider()


@st.cache_data(ttl=10)
def load_predictions():
    response = requests.get(API_URL, timeout=15)
    response.raise_for_status()

    records = response.json()["predictions"]
    return pd.DataFrame(records)


# Fetch live data
try:
    df = load_predictions()

except requests.RequestException as exc:
    st.error(f"Unable to connect to the monitoring API: {exc}")
    st.info("Check whether the Docker API container is running.")
    st.stop()

@st.cache_data(ttl=10)
def load_api_metrics():
    response = requests.get(METRICS_API_URL, timeout=15)
    response.raise_for_status()
    return response.json()

if df.empty:
    st.warning("No prediction records available yet.")
    st.stop()


df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce",
    utc=True
)

total = len(df)
delayed = int((df["prediction"] == 1).sum())
on_time = int((df["prediction"] == 0).sum())
avg_probability = float(df["delay_probability"].mean())

# API Performance Monitoring
st.subheader("API Performance Monitoring")

try:
    api_metrics = load_api_metrics()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total API Requests",
        f"{api_metrics['total_requests']:,}"
    )

    col2.metric(
        "Successful Requests",
        f"{api_metrics['successful_requests']:,}"
    )

    col3.metric(
        "Failed Requests",
        f"{api_metrics['error_requests']:,}"
    )

    col4.metric(
        "Error Rate",
        f"{api_metrics['error_rate']:.2%}"
    )

    col5, col6, col7 = st.columns(3)

    col5.metric(
        "Average Response Time",
        f"{api_metrics['average_response_time_ms']:.2f} ms"
    )

    col6.metric(
        "Minimum Response Time",
        f"{api_metrics['min_response_time_ms']:.2f} ms"
        if api_metrics["min_response_time_ms"] is not None
        else "N/A"
    )

    col7.metric(
        "Maximum Response Time",
        f"{api_metrics['max_response_time_ms']:.2f} ms"
    )

except requests.RequestException as exc:
    st.warning(f"Unable to load API performance metrics: {exc}")

st.caption(
    "API metrics are process-local counters and reset when the API restarts."
)

st.divider()

# Prediction overview
st.subheader("Prediction Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Predictions", f"{total:,}")
col2.metric("Predicted Delayed", f"{delayed:,}")
col3.metric("Predicted On Time", f"{on_time:,}")
col4.metric("Average Delay Probability", f"{avg_probability:.1%}")

st.divider()


# Prediction distribution
st.subheader("Prediction Distribution")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    distribution = pd.DataFrame({
        "Prediction": ["Predicted On Time", "Predicted Delayed"],
        "Count": [on_time, delayed]
    }).set_index("Prediction")

    st.bar_chart(distribution)

with chart_col2:
    st.subheader("Prediction Probability")
    st.metric("Average Probability", f"{avg_probability:.2%}")
    st.caption("Average model-estimated probability of delivery delay.")

st.divider()


# Prediction history
st.subheader("Prediction History")

history = df.sort_values("timestamp").set_index("timestamp")

st.line_chart(
    history[["delay_probability"]],
    y_label="Delay probability"
)

st.divider()


# Model information
st.subheader("Model Information")

model_col1, model_col2 = st.columns(2)

model_col1.write(f"**Model:** {df['model_name'].iloc[0]}")

versions = sorted(df["model_version"].dropna().unique().tolist())
model_col2.write(
    "**Model versions:** " + ", ".join(map(str, versions))
)

st.divider()

# Historical Model Performance
st.subheader("Historical Model Performance")

if TEST_EVALUATION_PATH.exists():

    try:
        evaluation = json.loads(
            TEST_EVALUATION_PATH.read_text(encoding="utf-8")
        )

        st.info(
            "Historical held-out test evaluation. "
            "These metrics are not live production performance."
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("PR-AUC", f"{evaluation['pr_auc']:.4f}")
        col2.metric("ROC-AUC", f"{evaluation['roc_auc']:.4f}")
        col3.metric("Precision", f"{evaluation['precision']:.2%}")
        col4.metric("Recall", f"{evaluation['recall']:.2%}")

        col5, col6, col7 = st.columns(3)

        col5.metric("F1 Score", f"{evaluation['f1']:.4f}")
        col6.metric("Test Samples", f"{evaluation['test_samples']:,}")
        col7.metric("Threshold", evaluation["threshold"])

        st.write(
            f"**Model:** {evaluation['model_name']} "
            f"| **Version:** {evaluation['model_version']}"
        )

        st.subheader("Test Confusion Matrix")

        cm = evaluation["confusion_matrix"]

        confusion_df = pd.DataFrame(
            cm,
            index=["Actual On Time", "Actual Delayed"],
            columns=["Predicted On Time", "Predicted Delayed"]
        )

        st.dataframe(confusion_df, width="stretch")

        with st.expander("Detailed Classification Report"):
            report_df = pd.DataFrame(
                evaluation["classification_report"]
            ).T

            st.dataframe(report_df, width="stretch")

    except (json.JSONDecodeError, KeyError, OSError, TypeError, ValueError) as exc:
        st.error(f"Unable to load model evaluation: {exc}")

else:
    st.warning("No historical model evaluation report found.")

st.divider()

# Data drift monitoring
st.subheader("Data Drift Monitoring")

if DRIFT_ALERT_PATH.exists():

    try:
        alert_data = json.loads(
            DRIFT_ALERT_PATH.read_text(encoding="utf-8")
        )

        status = alert_data.get("overall_status", "UNKNOWN")

        if status == "DRIFT_DETECTED":
            st.error("⚠️ Data Drift Detected")
        elif status == "NO_DRIFT":
            st.success("✅ No Significant Drift Detected")
        else:
            st.warning(f"Unknown drift status: {status}")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Features Checked",
            alert_data.get("features_checked", 0)
        )

        col2.metric(
            "Features with Drift",
            alert_data.get("features_with_drift", 0)
        )

        col3.metric(
            "Drift Threshold",
            alert_data.get("threshold", 0)
        )

        st.caption(
            f"Last analysis: {alert_data.get('timestamp', 'Unknown')}"
        )

        scores = alert_data.get("feature_scores", {})

        if scores:
            st.subheader("Feature Drift Scores")

            score_df = pd.DataFrame(
                list(scores.items()),
                columns=["Feature", "Drift Score"]
            ).sort_values("Drift Score", ascending=False)

            st.bar_chart(
                score_df.set_index("Feature")
            )

        alerts = alert_data.get("alerts", [])

        if alerts:
            st.subheader("Detected Drift Alerts")
            st.dataframe(
                pd.DataFrame(alerts),
                width="stretch"
            )

    except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
        st.error(f"Unable to read drift alerts: {exc}")

else:
    st.warning("No drift alert data found. Run the drift monitoring script first.")

if DRIFT_REPORT_PATH.exists():
    with st.expander("View Detailed Evidently HTML Report"):
        report_html = DRIFT_REPORT_PATH.read_text(encoding="utf-8")
        st.html(report_html)

st.info(
    "Drift alerts are based on the latest completed analysis. "
    "Current prediction records include generated test traffic, "
    "not verified production traffic."
)


# Recent predictions
st.subheader("Recent Predictions")

display_df = df.sort_values(
    "timestamp",
    ascending=False
).head(20).copy()

display_df["prediction"] = display_df["prediction"].map({
    0: "On Time",
    1: "Delayed"
})

display_df["delay_probability"] = (
    display_df["delay_probability"].map(
        lambda value: f"{value:.2%}"
    )
)

st.dataframe(display_df, width="stretch")

st.caption(
    "Predictions are model outputs, not verified delivery outcomes. "
    "The current records include generated test traffic."
)


# Sidebar
with st.sidebar:
    st.header("Dashboard Controls")

    if st.button("🔄 Refresh Dashboard"):
        st.cache_data.clear()
        st.rerun()

    st.caption("Data refresh interval: 10 seconds.")
    st.caption("Data source: Live FastAPI endpoint")
    st.code(API_URL)
