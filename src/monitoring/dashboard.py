
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
ALERTS_API_URL = "http://localhost:8000/monitoring/alerts?limit=100"

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
@st.cache_data(ttl=10)
def load_metrics_history():
    response = requests.get(
        "http://localhost:8000/monitoring/metrics/history?limit=500",
        timeout=15
    )
    response.raise_for_status()

    records = response.json()["metrics"]
    return pd.DataFrame(records)


@st.cache_data(ttl=10)
def load_performance_alerts():
    response = requests.get(ALERTS_API_URL, timeout=15)
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
# Persistent API Performance Alerts
st.subheader("API Performance Alerts")

try:
    alert_data = load_performance_alerts()
    alerts = alert_data.get("alerts", [])

    alerts_df = pd.DataFrame(alerts)

    if not alerts_df.empty:
        total_alerts = len(alerts_df)

        http_errors = int(
            (alerts_df["alert_type"] == "HTTP_ERROR").sum()
        )

        high_latency = int(
            (alerts_df["alert_type"] == "HIGH_LATENCY").sum()
        )

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Alerts", total_alerts)
        col2.metric("HTTP Errors", http_errors)
        col3.metric("High Latency Alerts", high_latency)

        st.markdown("**Recent Alert History**")

        alerts_df["timestamp"] = pd.to_datetime(
            alerts_df["timestamp"],
            errors="coerce",
            utc=True
        )

        alerts_df = alerts_df.sort_values(
            "timestamp",
            ascending=False
        )

        display_alerts = alerts_df[
            [
                "timestamp",
                "alert_type",
                "message",
                "endpoint",
                "response_time_ms",
                "status_code",
            ]
        ].copy()

        display_alerts["response_time_ms"] = (
            display_alerts["response_time_ms"].round(2)
        )

        display_alerts = display_alerts.rename(columns={
            "timestamp": "Time",
            "alert_type": "Alert Type",
            "message": "Message",
            "endpoint": "Endpoint",
            "response_time_ms": "Response (ms)",
            "status_code": "HTTP Status",
        })

        st.dataframe(
            display_alerts,
            width="stretch",
            hide_index=True,
        )

    else:
        st.success("No API performance alerts recorded.")

except requests.RequestException as exc:
    st.warning(f"Unable to load API performance alerts: {exc}")

# Historical API Performance
st.subheader("Historical API Performance")

try:
    history_df = load_metrics_history()

    if history_df.empty:
        st.info("No historical API performance records available yet.")
    else:
        history_df["timestamp"] = pd.to_datetime(
            history_df["timestamp"],
            errors="coerce",
            utc=True
        )

        history_df = history_df.dropna(subset=["timestamp"])
        history_df = history_df.sort_values("timestamp")

        # Request volume over time
        st.markdown("**Request Volume Over Time**")

        requests_over_time = (
            history_df.set_index("timestamp")
            .resample("1min")
            .size()
        )

        st.line_chart(
            requests_over_time,
            y_label="Number of requests"
        )

        # Response time over time
        st.markdown("**Response Time Over Time**")

        latency_over_time = history_df.set_index("timestamp")[
            ["response_time_ms"]
        ]

        st.line_chart(
            latency_over_time,
            y_label="Response time (ms)"
        )

        # Endpoint-wise performance
        st.markdown("**Endpoint-wise Performance**")

        endpoint_summary = (
            history_df.groupby("path")
            .agg(
                request_count=("path", "count"),
                average_response_time_ms=("response_time_ms", "mean"),
                max_response_time_ms=("response_time_ms", "max"),
                error_count=("status_code", lambda values: (values >= 400).sum())
            )
            .round(2)
            .sort_values("average_response_time_ms", ascending=False)
        )

        st.dataframe(endpoint_summary, width="stretch")
                # API Latency Alerts
        st.markdown("**API Latency Alerts**")

        LATENCY_THRESHOLD_MS = 500

        slow_endpoints = endpoint_summary[
            endpoint_summary["average_response_time_ms"]
            > LATENCY_THRESHOLD_MS
        ]

        if not slow_endpoints.empty:
            st.warning(
                f"{len(slow_endpoints)} endpoint(s) exceeded "
                f"the {LATENCY_THRESHOLD_MS} ms average latency threshold."
            )
            st.dataframe(slow_endpoints, width="stretch")
        else:
            st.success(
                f"All endpoints are below the "
                f"{LATENCY_THRESHOLD_MS} ms average latency threshold."
            )
                # HTTP Status Code Distribution
        st.markdown("**HTTP Status Code Distribution**")

        status_distribution = (
            history_df["status_code"]
            .value_counts()
            .sort_index()
            .rename_axis("HTTP Status")
            .to_frame("Request Count")
        )

        st.bar_chart(status_distribution)

        error_count = int(
            (history_df["status_code"] >= 400).sum()
        )

        if error_count > 0:
            st.warning(
                f"{error_count} failed HTTP requests detected."
            )
        else:
            st.success(
                "No failed HTTP requests in the recorded history."
            )



except requests.RequestException as exc:
    st.warning(f"Unable to load historical API metrics: {exc}")

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
