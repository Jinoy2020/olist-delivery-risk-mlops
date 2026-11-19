
from pathlib import Path
import os
import json

import pandas as pd
import requests
import streamlit as st


# ---------------------------------------------------------
# PROJECT CONFIGURATION
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# AWS EC2 API address.
# You can override this using the API_BASE_URL environment variable.
API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://16.4.43.214:8000"
).rstrip("/")

API_URL = f"{API_BASE_URL}/monitoring/predictions?limit=1000"
METRICS_API_URL = f"{API_BASE_URL}/monitoring/metrics"
ALERTS_API_URL = f"{API_BASE_URL}/monitoring/alerts?limit=100"
METRICS_HISTORY_API_URL = (
    f"{API_BASE_URL}/monitoring/metrics/history?limit=500"
)

DRIFT_STATUS_API_URL = f"{API_BASE_URL}/monitoring/drift"
DRIFT_REPORT_API_URL = f"{API_BASE_URL}/monitoring/drift/report"

TEST_EVALUATION_PATH = PROJECT_ROOT / "reports" / "test_evaluation.json"

LATENCY_THRESHOLD_MS = 500


# ---------------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Olist ML Monitoring",
    page_icon="📦",
    layout="wide"
)

st.title("📦 Olist Delivery Risk — Monitoring Dashboard")
st.caption("Live prediction and API monitoring through FastAPI.")

st.caption(f"API endpoint: {API_BASE_URL}")

st.divider()


# ---------------------------------------------------------
# API DATA LOADING FUNCTIONS
# ---------------------------------------------------------

@st.cache_data(ttl=10)
def load_predictions():
    response = requests.get(
        API_URL,
        timeout=15
    )
    response.raise_for_status()

    records = response.json().get("predictions", [])
    return pd.DataFrame(records)


@st.cache_data(ttl=10)
def load_api_metrics():
    response = requests.get(
        METRICS_API_URL,
        timeout=15
    )
    response.raise_for_status()

    return response.json()


@st.cache_data(ttl=10)
def load_metrics_history():
    response = requests.get(
        METRICS_HISTORY_API_URL,
        timeout=15
    )
    response.raise_for_status()

    records = response.json().get("metrics", [])
    return pd.DataFrame(records)


@st.cache_data(ttl=10)
def load_performance_alerts():
    response = requests.get(
        ALERTS_API_URL,
        timeout=15
    )
    response.raise_for_status()

    return response.json()
@st.cache_data(ttl=10)
def load_drift_status():
    response = requests.get(
        DRIFT_STATUS_API_URL,
        timeout=15
    )
    response.raise_for_status()

    return response.json()

# ---------------------------------------------------------
# LOAD LIVE PREDICTIONS
# ---------------------------------------------------------

try:
    df = load_predictions()

except requests.RequestException as exc:
    st.error(f"Unable to connect to the monitoring API: {exc}")
    st.info(
        "Check whether the AWS EC2 instance is running, "
        "the Docker API container is healthy, and port 8000 "
        "is allowed by your security group."
    )
    st.stop()


if df.empty:
    st.warning("No prediction records available yet.")
    st.stop()


required_prediction_columns = [
    "timestamp",
    "prediction",
    "delay_probability",
    "model_name",
    "model_version"
]

missing_columns = [
    column
    for column in required_prediction_columns
    if column not in df.columns
]

if missing_columns:
    st.error(
        "The prediction API response is missing required columns: "
        + ", ".join(missing_columns)
    )
    st.stop()


df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce",
    utc=True
)

df["prediction"] = pd.to_numeric(
    df["prediction"],
    errors="coerce"
)

df["delay_probability"] = pd.to_numeric(
    df["delay_probability"],
    errors="coerce"
)

df = df.dropna(
    subset=["timestamp", "prediction", "delay_probability"]
)

if df.empty:
    st.warning("No valid prediction records are available.")
    st.stop()


total = len(df)
delayed = int((df["prediction"] == 1).sum())
on_time = int((df["prediction"] == 0).sum())
avg_probability = float(df["delay_probability"].mean())


# ---------------------------------------------------------
# API PERFORMANCE MONITORING
# ---------------------------------------------------------

st.subheader("API Performance Monitoring")

try:
    api_metrics = load_api_metrics()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total API Requests",
        f"{api_metrics.get('total_requests', 0):,}"
    )

    col2.metric(
        "Successful Requests",
        f"{api_metrics.get('successful_requests', 0):,}"
    )

    col3.metric(
        "Failed Requests",
        f"{api_metrics.get('error_requests', 0):,}"
    )

    col4.metric(
        "Error Rate",
        f"{api_metrics.get('error_rate', 0):.2%}"
    )

    col5, col6, col7 = st.columns(3)

    average_time = api_metrics.get("average_response_time_ms")
    minimum_time = api_metrics.get("min_response_time_ms")
    maximum_time = api_metrics.get("max_response_time_ms")

    col5.metric(
        "Average Response Time",
        f"{average_time:.2f} ms"
        if average_time is not None
        else "N/A"
    )

    col6.metric(
        "Minimum Response Time",
        f"{minimum_time:.2f} ms"
        if minimum_time is not None
        else "N/A"
    )

    col7.metric(
        "Maximum Response Time",
        f"{maximum_time:.2f} ms"
        if maximum_time is not None
        else "N/A"
    )

except requests.RequestException as exc:
    st.warning(f"Unable to load API performance metrics: {exc}")

st.caption(
    "API metrics are process-local counters and reset when the API restarts."
)


# ---------------------------------------------------------
# PERSISTENT API PERFORMANCE ALERTS
# ---------------------------------------------------------

st.subheader("API Performance Alerts")

try:
    alert_data = load_performance_alerts()
    alerts = alert_data.get("alerts", [])

    alerts_df = pd.DataFrame(alerts)

    if not alerts_df.empty:

        total_alerts = len(alerts_df)

        if "alert_type" in alerts_df.columns:
            http_errors = int(
                (alerts_df["alert_type"] == "HTTP_ERROR").sum()
            )

            high_latency = int(
                (alerts_df["alert_type"] == "HIGH_LATENCY").sum()
            )
        else:
            http_errors = 0
            high_latency = 0

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Alerts", total_alerts)
        col2.metric("HTTP Errors", http_errors)
        col3.metric("High Latency Alerts", high_latency)

        st.markdown("**Recent Alert History**")

        if "timestamp" in alerts_df.columns:
            alerts_df["timestamp"] = pd.to_datetime(
                alerts_df["timestamp"],
                errors="coerce",
                utc=True
            )

            alerts_df = alerts_df.sort_values(
                "timestamp",
                ascending=False
            )

        display_columns = [
            "timestamp",
            "alert_type",
            "message",
            "endpoint",
            "response_time_ms",
            "status_code"
        ]

        available_columns = [
            column
            for column in display_columns
            if column in alerts_df.columns
        ]

        display_alerts = alerts_df[available_columns].copy()

        if "response_time_ms" in display_alerts.columns:
            display_alerts["response_time_ms"] = (
                pd.to_numeric(
                    display_alerts["response_time_ms"],
                    errors="coerce"
                ).round(2)
            )

        display_alerts = display_alerts.rename(columns={
            "timestamp": "Time",
            "alert_type": "Alert Type",
            "message": "Message",
            "endpoint": "Endpoint",
            "response_time_ms": "Response (ms)",
            "status_code": "HTTP Status"
        })

        st.dataframe(
            display_alerts,
            width="stretch",
            hide_index=True
        )

    else:
        st.success("No API performance alerts recorded.")

except requests.RequestException as exc:
    st.warning(f"Unable to load API performance alerts: {exc}")


# ---------------------------------------------------------
# HISTORICAL API PERFORMANCE
# ---------------------------------------------------------

st.subheader("Historical API Performance")

try:
    history_df = load_metrics_history()

    if history_df.empty:
        st.info(
            "No historical API performance records available yet."
        )

    else:
        required_history_columns = [
            "timestamp",
            "response_time_ms",
            "path",
            "status_code"
        ]

        missing_history_columns = [
            column
            for column in required_history_columns
            if column not in history_df.columns
        ]

        if missing_history_columns:
            st.warning(
                "Historical metrics are missing columns: "
                + ", ".join(missing_history_columns)
            )

        else:
            history_df["timestamp"] = pd.to_datetime(
                history_df["timestamp"],
                errors="coerce",
                utc=True
            )

            history_df["response_time_ms"] = pd.to_numeric(
                history_df["response_time_ms"],
                errors="coerce"
            )

            history_df["status_code"] = pd.to_numeric(
                history_df["status_code"],
                errors="coerce"
            )

            history_df = history_df.dropna(
                subset=[
                    "timestamp",
                    "response_time_ms",
                    "status_code"
                ]
            )

            history_df = history_df.sort_values("timestamp")

            if history_df.empty:
                st.info("No valid historical metrics are available.")

            else:
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

                latency_over_time = (
                    history_df.set_index("timestamp")[
                        ["response_time_ms"]
                    ]
                )

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
                        average_response_time_ms=(
                            "response_time_ms",
                            "mean"
                        ),
                        max_response_time_ms=(
                            "response_time_ms",
                            "max"
                        ),
                        error_count=(
                            "status_code",
                            lambda values: (values >= 400).sum()
                        )
                    )
                    .round(2)
                    .sort_values(
                        "average_response_time_ms",
                        ascending=False
                    )
                )

                st.dataframe(
                    endpoint_summary,
                    width="stretch"
                )

                # API Latency Alerts
                st.markdown("**API Latency Alerts**")

                slow_endpoints = endpoint_summary[
                    endpoint_summary["average_response_time_ms"]
                    > LATENCY_THRESHOLD_MS
                ]

                if not slow_endpoints.empty:
                    st.warning(
                        f"{len(slow_endpoints)} endpoint(s) exceeded "
                        f"the {LATENCY_THRESHOLD_MS} ms average "
                        "latency threshold."
                    )

                    st.dataframe(
                        slow_endpoints,
                        width="stretch"
                    )

                else:
                    st.success(
                        "All endpoints are below the "
                        f"{LATENCY_THRESHOLD_MS} ms average "
                        "latency threshold."
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
    st.warning(
        f"Unable to load historical API metrics: {exc}"
    )


st.divider()


# ---------------------------------------------------------
# PREDICTION OVERVIEW
# ---------------------------------------------------------

st.subheader("Prediction Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Predictions", f"{total:,}")
col2.metric("Predicted Delayed", f"{delayed:,}")
col3.metric("Predicted On Time", f"{on_time:,}")
col4.metric(
    "Average Delay Probability",
    f"{avg_probability:.1%}"
)

st.divider()


# ---------------------------------------------------------
# PREDICTION DISTRIBUTION
# ---------------------------------------------------------

st.subheader("Prediction Distribution")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:

    distribution = pd.DataFrame({
        "Prediction": [
            "Predicted On Time",
            "Predicted Delayed"
        ],
        "Count": [
            on_time,
            delayed
        ]
    }).set_index("Prediction")

    st.bar_chart(distribution)

with chart_col2:

    st.subheader("Prediction Probability")

    st.metric(
        "Average Probability",
        f"{avg_probability:.2%}"
    )

    st.caption(
        "Average model-estimated probability of delivery delay."
    )

st.divider()


# ---------------------------------------------------------
# PREDICTION HISTORY
# ---------------------------------------------------------

st.subheader("Prediction History")

history = (
    df.sort_values("timestamp")
    .set_index("timestamp")
)

st.line_chart(
    history[["delay_probability"]],
    y_label="Delay probability"
)

st.divider()


# ---------------------------------------------------------
# MODEL INFORMATION
# ---------------------------------------------------------

st.subheader("Model Information")

model_col1, model_col2 = st.columns(2)

model_col1.write(
    f"**Model:** {df['model_name'].iloc[0]}"
)

versions = sorted(
    df["model_version"].dropna().unique().tolist(),
    key=str
)

model_col2.write(
    "**Model versions:** " + ", ".join(map(str, versions))
)

st.divider()


# ---------------------------------------------------------
# HISTORICAL MODEL PERFORMANCE
# ---------------------------------------------------------

st.subheader("Historical Model Performance")

if TEST_EVALUATION_PATH.exists():

    try:
        evaluation = json.loads(
            TEST_EVALUATION_PATH.read_text(
                encoding="utf-8"
            )
        )

        st.info(
            "Historical held-out test evaluation. "
            "These metrics are not live production performance."
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "PR-AUC",
            f"{evaluation['pr_auc']:.4f}"
        )

        col2.metric(
            "ROC-AUC",
            f"{evaluation['roc_auc']:.4f}"
        )

        col3.metric(
            "Precision",
            f"{evaluation['precision']:.2%}"
        )

        col4.metric(
            "Recall",
            f"{evaluation['recall']:.2%}"
        )

        col5, col6, col7 = st.columns(3)

        col5.metric(
            "F1 Score",
            f"{evaluation['f1']:.4f}"
        )

        col6.metric(
            "Test Samples",
            f"{evaluation['test_samples']:,}"
        )

        col7.metric(
            "Threshold",
            evaluation["threshold"]
        )

        st.write(
            f"**Model:** {evaluation['model_name']} "
            f"| **Version:** {evaluation['model_version']}"
        )

        st.subheader("Test Confusion Matrix")

        cm = evaluation["confusion_matrix"]

        confusion_df = pd.DataFrame(
            cm,
            index=[
                "Actual On Time",
                "Actual Delayed"
            ],
            columns=[
                "Predicted On Time",
                "Predicted Delayed"
            ]
        )

        st.dataframe(
            confusion_df,
            width="stretch"
        )

        with st.expander("Detailed Classification Report"):

            report_df = pd.DataFrame(
                evaluation["classification_report"]
            ).T

            st.dataframe(
                report_df,
                width="stretch"
            )

    except (
        json.JSONDecodeError,
        KeyError,
        OSError,
        TypeError,
        ValueError
    ) as exc:

        st.error(
            f"Unable to load model evaluation: {exc}"
        )

else:
    st.warning(
        "No historical model evaluation report found."
    )


st.divider()

# ---------------------------------------------------------
# DATA DRIFT MONITORING
# ---------------------------------------------------------

st.subheader("Data Drift Monitoring")

try:
    alert_data = load_drift_status()

    status = alert_data.get("overall_status", "UNKNOWN")

    if status == "INSUFFICIENT_DATA":

        st.warning("Insufficient Production Data")

        st.info(
            alert_data.get(
                "message",
                "Waiting for sufficient production prediction records."
            )
        )

    elif status == "INSUFFICIENT_METRICS":

        st.warning("Insufficient Drift Metrics")

    elif status == "DRIFT_DETECTED":

        st.error("Data Drift Detected")

    elif status == "NO_DRIFT":

        st.success("No Significant Drift Detected")

    elif status == "NOT_AVAILABLE":

        st.info("Drift Monitoring Not Available")

    else:

        st.warning(f"Unknown drift status: {status}")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Current Production Records",
        alert_data.get("current_records", 0)
    )

    col2.metric(
        "Minimum Required Records",
        alert_data.get("minimum_records", 100)
    )

    col3.metric(
        "Drift Threshold",
        alert_data.get("threshold", 0)
    )

    st.caption(
        "Latest monitoring update: "
        f"{alert_data.get('timestamp', 'Unknown')}"
    )

    # Last successful analysis

    previous_analysis = alert_data.get(
        "last_successful_analysis"
    )

    if previous_analysis:

        st.divider()

        st.subheader("Last Successful Drift Analysis")

        st.caption(
            "Analysis timestamp: "
            f"{previous_analysis.get('timestamp', 'Unknown')}"
        )

        col1, col2 = st.columns(2)

        col1.metric(
            "Features Checked",
            previous_analysis.get("features_checked", 0)
        )

        col2.metric(
            "Features with Drift",
            previous_analysis.get("features_with_drift", 0)
        )

        scores = previous_analysis.get("feature_scores", {})

        if scores:

            st.subheader("Historical Feature Drift Scores")

            score_df = pd.DataFrame(
                list(scores.items()),
                columns=["Feature", "Drift Score"]
            ).sort_values(
                "Drift Score",
                ascending=False
            )

            st.bar_chart(
                score_df.set_index("Feature")
            )

        historical_alerts = previous_analysis.get("alerts", [])

        if historical_alerts:

            st.subheader("Historical Drift Alerts")

            st.dataframe(
                pd.DataFrame(historical_alerts),
                width="stretch",
                hide_index=True
            )

    else:

        st.info(
            "No successful drift analysis is available yet."
        )

    # Evidently HTML report hosted by FastAPI

    st.divider()

    st.subheader("Evidently Drift Report")

    st.caption(
        "Open the latest HTML drift report served directly by the EC2 API."
    )

    st.link_button(
        "View Evidently HTML Report",
        DRIFT_REPORT_API_URL
    )

except requests.RequestException as exc:

    st.error(
        f"Unable to load drift monitoring status: {exc}"
    )

st.info(
    "Drift detection uses production-labelled prediction records only. "
    "Generated test traffic is excluded from production drift analysis."
)

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.header("Dashboard Controls")

    if st.button("🔄 Refresh Dashboard"):

        st.cache_data.clear()
        st.rerun()

    st.caption(
        "Data refresh interval: 10 seconds."
    )

    st.caption(
        "Data source: Live FastAPI endpoint"
    )

    st.code(API_URL)
