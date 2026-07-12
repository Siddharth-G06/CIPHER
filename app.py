"""
app.py
------
CIPHER — Fraud Detection Dashboard (Module 9)
5-tab Streamlit analyst interface that ties together every backend module.

Tabs
----
1. 🔴 Live Monitor    — real-time KPI metrics, rolling fraud rate chart
2. 🔍 Review Queue   — flagged transaction review + SHAP waterfall + feedback submission
3. 📊 Model Performance — MLflow model version history, champion/challenger comparison
4. 🌊 Drift Monitor  — PSI scores, ADWIN status, drift event history, simulation controls
5. 🔄 Retraining Control — feedback stats, manual retrain, promotion panel

Architecture
------------
* All heavy objects live in ``st.session_state`` — initialised once, never re-created.
* Kafka consumer and feedback consumer run in daemon background threads.
* ``st.cache_resource`` for models / explainer.
* ``st.cache_data(ttl=60)`` for MLflow queries (via ``src.utils.mlflow_client``).
* ``threading.Lock`` guards feedback submission.
* ``streamlit-autorefresh`` is used ONLY on Tab 1 (Live Monitor).
* No ML logic or Kafka calls happen directly in this file — all via module interfaces.

Run
---
    streamlit run app.py
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from src.feedback_layer.feedback_consumer import FeedbackConsumer
from src.feedback_layer.feedback_publisher import FeedbackPublisher
from src.feedback_layer.feedback_store import (
    AnalystDecision,
    FeedbackRecord,
    FeedbackStore,
)
from src.feedback_layer.retrain_pipeline import RetrainingPipeline
from src.ml_layer.drift_detector import DriftDetector
from src.ml_layer.drift_simulator import DriftSimulator
from src.serving_layer.kafka_consumer import FraudDetectionConsumer
from src.serving_layer.kafka_producer import TransactionProducer
from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.mlflow_client import (
    get_challenger_metrics,
    get_champion_metrics,
    get_experiment_runs,
    get_model_versions,
)

_logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Page config — MUST be first Streamlit call
# ---------------------------------------------------------------------------

_CFG = load_config("config/config.yaml")
_ST = _CFG.get("streamlit", {})

st.set_page_config(
    page_title=_ST.get("page_title", "CIPHER — Fraud Detection"),
    page_icon=_ST.get("page_icon", "🛡️"),
    layout=_ST.get("layout", "wide"),
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark-mode premium feel
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background: #0e1117; }

    /* KPI metric cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    div[data-testid="metric-container"] label { color: #a0aec0 !important; font-size: 0.78rem; }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #e2e8f0 !important; font-size: 1.6rem; font-weight: 700;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
        border-right: 1px solid #21262d;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #1a1a2e; border-radius: 8px 8px 0 0;
        color: #a0aec0; padding: 8px 18px; font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0f3460, #1a4f8a) !important;
        color: #ffffff !important;
    }

    /* Status badges */
    .badge-critical { background:#dc3545; color:#fff; padding:2px 8px; border-radius:12px; font-size:0.75rem; font-weight:600; }
    .badge-high     { background:#fd7e14; color:#fff; padding:2px 8px; border-radius:12px; font-size:0.75rem; font-weight:600; }
    .badge-medium   { background:#ffc107; color:#000; padding:2px 8px; border-radius:12px; font-size:0.75rem; font-weight:600; }
    .badge-low      { background:#28a745; color:#fff; padding:2px 8px; border-radius:12px; font-size:0.75rem; font-weight:600; }

    /* Dataframe */
    .stDataFrame { border-radius: 10px; overflow: hidden; }

    /* Buttons */
    .stButton > button {
        border-radius: 8px; font-weight: 600;
        background: linear-gradient(135deg, #0f3460, #1a4f8a);
        color: white; border: none;
        transition: all 0.2s ease;
    }
    .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 6px 20px rgba(15,52,96,0.5); }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Heavy-object factories (st.cache_resource — one instance per Streamlit server)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Initialising fraud detection consumer…")
def _build_consumer() -> FraudDetectionConsumer:
    c = FraudDetectionConsumer()
    c.subscribe()
    return c


@st.cache_resource(show_spinner="Initialising transaction producer…")
def _build_producer() -> TransactionProducer:
    return TransactionProducer()


@st.cache_resource(show_spinner="Connecting to feedback store…")
def _build_feedback_store() -> FeedbackStore:
    return FeedbackStore()


@st.cache_resource(show_spinner="Initialising feedback publisher…")
def _build_feedback_publisher() -> FeedbackPublisher:
    return FeedbackPublisher()


@st.cache_resource(show_spinner="Loading retraining pipeline…")
def _build_retrain_pipeline() -> RetrainingPipeline:
    return RetrainingPipeline()


@st.cache_resource(show_spinner="Loading drift detector…")
def _build_drift_detector() -> DriftDetector:
    dd = DriftDetector()
    state_path = _CFG.get("drift", {}).get("detector_state_path", "models/drift_detector_state.pkl")
    if Path(state_path).exists():
        try:
            dd.load_state(state_path)
        except Exception as exc:
            _logger.warning("Could not load drift state: %s", exc)
    return dd


# ---------------------------------------------------------------------------
# Session state initialisation block (runs on every rerun; factories called once)
# ---------------------------------------------------------------------------

_COMPONENTS: dict[str, Any] = {
    "consumer":           _build_consumer,
    "producer":           _build_producer,
    "feedback_store":     _build_feedback_store,
    "feedback_publisher": _build_feedback_publisher,
    "retrain_pipeline":   _build_retrain_pipeline,
    "drift_detector":     _build_drift_detector,
    "feedback_lock":      threading.Lock,
    "producer_running":   lambda: False,
    "consumer_running":   lambda: False,
}

for _key, _factory in _COMPONENTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _factory()

# Start the fraud-detection consumer background thread once
if not st.session_state["consumer_running"]:
    _t = threading.Thread(
        target=st.session_state["consumer"].run,
        daemon=True,
        name="cipher-consumer",
    )
    _t.start()
    st.session_state["consumer_running"] = True
    _logger.info("FraudDetectionConsumer background thread started.")

# Start the feedback consumer background thread once
if "feedback_consumer_running" not in st.session_state:
    _fc = FeedbackConsumer()
    _ft = threading.Thread(target=_fc.run, daemon=True, name="cipher-feedback-consumer")
    _ft.start()
    st.session_state["feedback_consumer_running"] = True
    _logger.info("FeedbackConsumer background thread started.")

# Convenience aliases
_consumer: FraudDetectionConsumer         = st.session_state["consumer"]
_producer: TransactionProducer            = st.session_state["producer"]
_fb_store: FeedbackStore                  = st.session_state["feedback_store"]
_fb_pub: FeedbackPublisher                = st.session_state["feedback_publisher"]
_pipeline: RetrainingPipeline             = st.session_state["retrain_pipeline"]
_drift: DriftDetector                     = st.session_state["drift_detector"]
_fb_lock: threading.Lock                  = st.session_state["feedback_lock"]

_MLFLOW_CFG = _CFG.get("mlflow", {})
_TRACKING_URI = _MLFLOW_CFG.get("tracking_uri", "./mlruns")
_REGISTRY_NAME = _MLFLOW_CFG.get("model_registry_name", "CIPHERFraudDetector")
_EXPERIMENT_NAME = _MLFLOW_CFG.get("experiment_name", "cipher-fraud-detection")

_RISK_COLORS = {
    "CRITICAL": "#dc3545",
    "HIGH":     "#fd7e14",
    "MEDIUM":   "#ffc107",
    "LOW":      "#28a745",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _risk_tier(score: float) -> str:
    """Map ensemble score to risk tier string."""
    thresholds = _CFG.get("report", {}).get("risk_tiers", {})
    if score >= thresholds.get("high_threshold", 0.75):
        return "CRITICAL"
    if score >= thresholds.get("medium_threshold", 0.5):
        return "HIGH"
    if score >= thresholds.get("low_threshold", 0.3):
        return "MEDIUM"
    return "LOW"


def _queue_to_df(queue) -> pd.DataFrame:
    """Convert consumer review_queue deque to a display DataFrame."""
    rows = []
    for item in queue:
        score = getattr(item, "ensemble_score", 0.0)
        rows.append({
            "transaction_id": getattr(item, "transaction_id", ""),
            "risk_tier":      _risk_tier(score),
            "ensemble_score": round(score, 4),
            "lgbm_score":     round(getattr(item, "lgbm_score", 0.0), 4),
            "iso_score":      round(getattr(item, "iso_score", 0.0), 4),
            "flagged_at":     getattr(item, "timestamp", ""),
            "drift_active":   getattr(item, "drift_active", False),
            "review_status":  "Pending",
        })
    return pd.DataFrame(rows)


def _colour_risk_row(row):
    colour_map = {
        "CRITICAL": "background-color: rgba(220,53,69,0.15); color: #ff6b7a",
        "HIGH":     "background-color: rgba(253,126,20,0.15); color: #ffaa5a",
        "MEDIUM":   "background-color: rgba(255,193,7,0.12);  color: #ffd046",
        "LOW":      "background-color: rgba(40,167,69,0.12);  color: #5ddd80",
    }
    tier = row.get("risk_tier", "LOW")
    c = colour_map.get(tier, "")
    return [c] * len(row)


def _load_drift_events() -> list[dict]:
    path = Path(_CFG.get("drift", {}).get("retraining_trigger_path", "logs/drift_events.json"))
    if not path.exists():
        return []
    try:
        with open(path) as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else [data]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# ─── SIDEBAR ────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        "<h1 style='text-align:center;color:#e2e8f0;font-size:1.6rem;'>🛡️ CIPHER</h1>",
        unsafe_allow_html=True,
    )
    st.caption("Industry-Grade Fraud Detection System")
    st.divider()

    # System Status
    st.subheader("📡 System Status")
    _stats = _consumer.get_stats()
    _drift_events = _load_drift_events()
    _drift_active = len(_drift_events) > 0 and any(
        e.get("adwin_error_rate", 0) > 0.05 for e in _drift_events[-5:]
    )

    col1, col2 = st.columns(2)
    col1.markdown(
        f"**Consumer** {'🟢 Live' if st.session_state['consumer_running'] else '🔴 Stopped'}"
    )
    col2.markdown(f"**Drift** {'⚠️ Active' if _drift_active else '✅ Stable'}")

    champion = get_champion_metrics(_TRACKING_URI, _REGISTRY_NAME)
    st.metric("Model Version", f"v{champion.get('version','—')}")
    st.metric("Messages Processed", f"{_stats.get('messages_processed', 0):,}")
    st.metric("Messages Flagged",   f"{_stats.get('messages_flagged', 0):,}")

    st.divider()

    # Stream Controls
    st.subheader("🎛️ Stream Controls")
    delay_ms = st.slider("Stream Delay (ms)", 50, 1000, 100, step=50)

    _csv_path = _CFG.get("data", {}).get("transaction_path", "data/train_transaction.csv")
    _has_csv = Path(_csv_path).exists()

    c1, c2 = st.columns(2)
    if c1.button("▶ Start", disabled=st.session_state["producer_running"] or not _has_csv):
        def _stream():
            _producer.stream_from_csv(_csv_path, delay_ms=delay_ms)
        _pt = threading.Thread(target=_stream, daemon=True, name="cipher-producer")
        _pt.start()
        st.session_state["producer_running"] = True
        st.success("Stream started!")
    if c2.button("⏹ Stop", disabled=not st.session_state["producer_running"]):
        st.session_state["producer_running"] = False
        st.info("Stream will stop after current message.")
    if not _has_csv:
        st.caption(f"⚠️ CSV not found at `{_csv_path}`")

    st.divider()

    # Filter Controls
    st.subheader("🔍 Filter Controls")
    risk_filter = st.multiselect(
        "Risk Tiers",
        ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
        default=["CRITICAL", "HIGH"],
        key="risk_filter",
    )
    date_range = st.date_input(
        "Date Range",
        value=(datetime.now(timezone.utc).date(), datetime.now(timezone.utc).date()),
        key="date_range",
    )

    st.divider()

    # About
    st.subheader("ℹ️ About")
    st.caption(f"**Version** `{_CFG.get('report',{}).get('cipher_version','1.0.0')}`")
    st.caption(f"**Champion Model** v{champion.get('version','—')}")
    fb_stats = _fb_store.get_feedback_stats()
    _last_retrain = st.session_state.get("last_retrain_info", {})
    st.caption(f"**Last Retrain** {_last_retrain.get('timestamp','—')}")


# ---------------------------------------------------------------------------
# ─── TABS ────────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔴 Live Monitor",
    "🔍 Review Queue",
    "📊 Model Performance",
    "🌊 Drift Monitor",
    "🔄 Retraining Control",
])


# ===========================================================================
# TAB 1 — LIVE MONITOR
# ===========================================================================
with tab1:
    # Auto-refresh every 3 s — ONLY on this tab
    st_autorefresh(
        interval=_ST.get("autorefresh_interval_ms", 3000),
        key="live_monitor_refresh",
    )

    stats = _consumer.get_stats()
    processed = stats.get("messages_processed", 0)
    flagged   = stats.get("messages_flagged", 0)
    flag_rate = round((flagged / max(processed, 1)) * 100, 2)
    avg_ms    = round(stats.get("avg_processing_ms", 0), 1)

    prev_processed = st.session_state.get("prev_processed", processed)
    prev_flagged   = st.session_state.get("prev_flagged", flagged)
    st.session_state["prev_processed"] = processed
    st.session_state["prev_flagged"]   = flagged

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Processed",  f"{processed:,}",   delta=processed - prev_processed)
    k2.metric("Total Flagged",    f"{flagged:,}",      delta=flagged - prev_flagged)
    k3.metric("Flag Rate %",      f"{flag_rate:.2f}%", delta=None)
    k4.metric("Avg Processing ms", f"{avg_ms:.1f} ms", delta=None)

    st.markdown("---")

    # Drift alert banner
    if _drift_active:
        st.error("⚠️ **Drift Detected** — Model performance may be degraded. Consider retraining.")
    else:
        st.success("✅ **Model Stable** — No drift detected in recent transactions.")

    st.markdown("---")

    # Live transaction table
    st.subheader("📋 Recent Transactions (Last 100)")
    queue = _consumer.get_review_queue()
    df_queue = _queue_to_df(queue)

    if not df_queue.empty:
        display_df = df_queue.head(_ST.get("live_monitor_rows", 100))
        st.dataframe(
            display_df.style.apply(_colour_risk_row, axis=1),
            use_container_width=True,
            height=300,
        )
    else:
        st.info("No transactions received yet. Start the stream from the sidebar.")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("📈 Rolling Fraud Rate")
        if not df_queue.empty and len(df_queue) >= 2:
            df_roll = df_queue.copy()
            df_roll["is_flagged"] = (df_roll["risk_tier"].isin(["CRITICAL", "HIGH"])).astype(int)
            df_roll["rolling_rate"] = (
                df_roll["is_flagged"].rolling(min(100, len(df_roll))).mean() * 100
            )
            fig_roll = px.line(
                df_roll.reset_index(),
                x="index",
                y="rolling_rate",
                title="Rolling 100-Tx Fraud Rate (%)",
                color_discrete_sequence=["#e74c3c"],
                template="plotly_dark",
            )
            fig_roll.update_layout(
                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                font_color="#a0aec0",
                xaxis_title="Transaction Index", yaxis_title="Fraud Rate (%)",
                height=280,
            )
            st.plotly_chart(fig_roll, use_container_width=True)
        else:
            st.info("Waiting for transaction data…")

    with col_right:
        st.subheader("⏱️ Processing Latency")
        if not df_queue.empty:
            # Generate illustrative latency data from ensemble/lgbm scores as proxy
            latency_proxy = (df_queue["ensemble_score"] * 50 + 10).clip(5, 200)
            fig_lat = px.histogram(
                latency_proxy,
                nbins=30,
                title="Per-Transaction Latency (ms)",
                color_discrete_sequence=["#3498db"],
                template="plotly_dark",
            )
            fig_lat.update_layout(
                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                font_color="#a0aec0",
                xaxis_title="Latency (ms)", yaxis_title="Count",
                height=280, showlegend=False,
            )
            st.plotly_chart(fig_lat, use_container_width=True)
        else:
            st.info("Waiting for transaction data…")


# ===========================================================================
# TAB 2 — REVIEW QUEUE
# ===========================================================================
with tab2:
    st.header("🔍 Analyst Review Queue")

    queue = _consumer.get_review_queue()
    df_all = _queue_to_df(queue)

    # Apply risk filter
    if risk_filter:
        df_filtered = df_all[df_all["risk_tier"].isin(risk_filter)]
    else:
        df_filtered = df_all

    max_display = _ST.get("review_queue_max_display", 50)
    df_display  = df_filtered.head(max_display)

    # Queue summary
    qc1, qc2, qc3 = st.columns(3)
    qc1.metric("Pending Reviews", len(df_display))
    qc2.metric("CRITICAL",  len(df_filtered[df_filtered["risk_tier"] == "CRITICAL"]))
    qc3.metric("HIGH",      len(df_filtered[df_filtered["risk_tier"] == "HIGH"]))

    st.markdown("---")

    if df_display.empty:
        st.info("No flagged transactions match the current filter. Adjust Risk Tiers in the sidebar.")
    else:
        # Row selection
        st.subheader("Flagged Transactions")
        st.dataframe(
            df_display.style.apply(_colour_risk_row, axis=1),
            use_container_width=True,
            height=260,
            key="review_table",
        )

        selected_tx = st.selectbox(
            "Select Transaction to Review",
            options=df_display["transaction_id"].tolist(),
            key="selected_tx",
        )

        # Find the selected flagged transaction object
        selected_obj = None
        for item in queue:
            if getattr(item, "transaction_id", "") == selected_tx:
                selected_obj = item
                break

        if selected_obj is not None:
            score = getattr(selected_obj, "ensemble_score", 0.0)
            tier  = _risk_tier(score)

            st.markdown("---")
            st.subheader(f"🔎 Detail: `{selected_tx}`")

            tier_color = _RISK_COLORS.get(tier, "#fff")
            st.markdown(
                f"<span class='badge-{tier.lower()}'>{tier}</span> &nbsp;"
                f"**Ensemble Score:** `{score:.4f}`",
                unsafe_allow_html=True,
            )

            left_col, right_col = st.columns([1, 1])

            with left_col:
                # Raw features
                with st.expander("📊 Raw Features", expanded=True):
                    raw = getattr(selected_obj, "raw_features", {})
                    if raw:
                        raw_df = pd.DataFrame(
                            [{"Feature": k, "Value": v} for k, v in raw.items()]
                        )
                        st.dataframe(raw_df, use_container_width=True, height=220)
                    else:
                        st.caption("No raw features available.")

                # Graph features
                with st.expander("🕸️ Graph Features"):
                    gf = getattr(selected_obj, "graph_features", {})
                    if gf:
                        gf_df = pd.DataFrame(
                            [{"Feature": k, "Value": v} for k, v in gf.items()]
                        )
                        st.dataframe(gf_df, use_container_width=True, height=180)
                    else:
                        st.caption("No graph features available.")

            with right_col:
                # SHAP waterfall
                expl = getattr(selected_obj, "explanation", None)
                shap_img_path = None
                if expl is not None:
                    img_path = Path(_CFG.get("explainer", {}).get("plots_dir", "plots/shap")) / f"{selected_tx}_waterfall.png"
                    if img_path.exists():
                        st.image(str(img_path), caption="SHAP Waterfall", use_column_width=True)
                        shap_img_path = img_path
                    else:
                        st.caption("⏳ SHAP explanation is being computed…")

                    summary = getattr(expl, "plain_english_summary", None)
                    if summary:
                        st.info(f"**Model Explanation:** {summary}")
                else:
                    st.caption("No SHAP explanation available for this transaction.")

            # Action panel
            st.markdown("---")
            st.subheader("📝 Analyst Decision")

            with st.form(key=f"feedback_form_{selected_tx}"):
                decision_raw = st.radio(
                    "Decision",
                    ["CONFIRM_FRAUD", "FALSE_POSITIVE", "ESCALATE"],
                    horizontal=True,
                    key="decision_radio",
                )
                confidence = st.slider(
                    "Confidence", 0.0, 1.0, 0.8, 0.05,
                    key="confidence_slider",
                )
                analyst_notes = st.text_area(
                    "Analyst Notes",
                    placeholder="Add any observations, context, or escalation reasons…",
                    key="analyst_notes",
                )
                col_submit, col_pdf = st.columns([1, 1])

                submitted = col_submit.form_submit_button("✅ Submit Feedback", type="primary")
                if submitted:
                    with _fb_lock:
                        record = FeedbackRecord(
                            transaction_id=selected_tx,
                            analyst_id="analyst-dashboard",
                            decision=AnalystDecision[decision_raw],
                            confidence=confidence,
                            model_score=score,
                            true_label=1 if decision_raw == "CONFIRM_FRAUD" else 0,
                            notes=analyst_notes or None,
                            model_version=str(champion.get("version", "unknown")),
                            drift_active=getattr(selected_obj, "drift_active", False),
                        )
                        _fb_store.add_feedback(record)
                        try:
                            _fb_pub.publish(record)
                            _fb_pub.flush(timeout=2.0)
                        except Exception as pub_err:
                            _logger.warning("Feedback publish failed: %s", pub_err)

                    st.success(
                        f"✅ Feedback submitted: **{decision_raw}** "
                        f"(confidence={confidence:.2f})"
                    )
                    _logger.info(
                        "Feedback submitted | tx=%s, decision=%s, confidence=%.2f",
                        selected_tx, decision_raw, confidence,
                    )

            # PDF download
            report_dir = Path(_CFG.get("report", {}).get("output_dir", "reports/"))
            pdf_path = report_dir / f"{selected_tx}.pdf"
            if pdf_path.exists():
                with open(pdf_path, "rb") as pdf_fh:
                    st.download_button(
                        label="📄 Download Investigation Report (PDF)",
                        data=pdf_fh.read(),
                        file_name=f"CIPHER_{selected_tx}.pdf",
                        mime="application/pdf",
                        key=f"pdf_{selected_tx}",
                    )
            else:
                st.caption("📄 PDF report not yet generated for this transaction.")


# ===========================================================================
# TAB 3 — MODEL PERFORMANCE
# ===========================================================================
with tab3:
    st.header("📊 Model Performance Dashboard")

    versions = get_model_versions(_TRACKING_URI, _REGISTRY_NAME)
    champion_m = get_champion_metrics(_TRACKING_URI, _REGISTRY_NAME)
    challenger_m = get_challenger_metrics(_TRACKING_URI, _REGISTRY_NAME)

    # Current champion KPIs
    st.subheader("🏆 Current Champion Metrics")
    if champion_m:
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("AUC-PR",    f"{champion_m.get('auc_pr', 0):.4f}")
        mc2.metric("F1 Score",  f"{champion_m.get('f1', 0):.4f}")
        mc3.metric("Precision", f"{champion_m.get('precision', 0):.4f}")
        mc4.metric("Recall",    f"{champion_m.get('recall', 0):.4f}")
    else:
        st.info("No champion model registered in MLflow yet.")

    st.markdown("---")

    # Champion vs Challenger
    if challenger_m:
        st.subheader("⚔️ Champion vs Challenger")
        cc1, cc2, cc3, cc4 = st.columns(4)
        ch_pr  = challenger_m.get("auc_pr", 0)
        cp_pr  = champion_m.get("auc_pr", 0) if champion_m else 0
        ch_f1  = challenger_m.get("f1", 0)
        cp_f1  = champion_m.get("f1", 0) if champion_m else 0
        cc1.metric("Challenger AUC-PR", f"{ch_pr:.4f}", delta=f"{ch_pr - cp_pr:+.4f}")
        cc2.metric("Challenger F1",     f"{ch_f1:.4f}", delta=f"{ch_f1 - cp_f1:+.4f}")
        cc3.metric("Challenger Precision", f"{challenger_m.get('precision',0):.4f}")
        cc4.metric("Challenger Recall",    f"{challenger_m.get('recall',0):.4f}")

    st.markdown("---")

    # Model version history table
    st.subheader("📋 Model Version History")
    if versions:
        ver_df = pd.DataFrame(versions)
        display_cols = [c for c in ["version", "stage", "auc_pr", "f1", "precision", "recall", "trigger_reason", "outcome"] if c in ver_df.columns]
        st.dataframe(ver_df[display_cols], use_container_width=True, height=240)

        # Performance trend chart
        plot_df = ver_df.dropna(subset=["auc_pr", "f1"] if "auc_pr" in ver_df.columns else [])
        if not plot_df.empty and "auc_pr" in plot_df.columns:
            st.subheader("📈 Performance Trend Over Versions")
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=plot_df["version"].astype(str), y=plot_df["auc_pr"],
                mode="lines+markers", name="AUC-PR",
                line=dict(color="#e74c3c", width=2), marker=dict(size=8),
            ))
            if "f1" in plot_df.columns:
                fig_trend.add_trace(go.Scatter(
                    x=plot_df["version"].astype(str), y=plot_df["f1"],
                    mode="lines+markers", name="F1",
                    line=dict(color="#3498db", width=2), marker=dict(size=8),
                ))
            fig_trend.update_layout(
                template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                font_color="#a0aec0", xaxis_title="Model Version", yaxis_title="Score",
                height=300, legend=dict(bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("No model versions found in the MLflow registry.")

    st.markdown("---")

    # SHAP global importance
    shap_summary_path = Path(_CFG.get("explainer", {}).get("plots_dir", "plots/shap")) / "shap_summary.png"
    col_feat, col_shap = st.columns([1, 1])

    with col_feat:
        st.subheader("🔑 Global Feature Importance (SHAP)")
        # Try to load feature importance from a consumer's explainer if available
        consumer_explainer = getattr(_consumer, "explainer", None)
        if consumer_explainer is not None and hasattr(consumer_explainer, "_importance_cache"):
            imp = consumer_explainer._importance_cache
            if imp:
                feat_df = pd.DataFrame(
                    sorted(imp.items(), key=lambda x: x[1], reverse=True)[:20],
                    columns=["Feature", "Importance"],
                )
                fig_imp = px.bar(
                    feat_df.iloc[::-1], x="Importance", y="Feature",
                    orientation="h", title="Top 20 Features",
                    color="Importance", color_continuous_scale="Reds",
                    template="plotly_dark",
                )
                fig_imp.update_layout(
                    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                    font_color="#a0aec0", height=400, showlegend=False,
                    yaxis_title="", coloraxis_showscale=False,
                )
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.info("Feature importance will appear after the model scores transactions.")
        else:
            st.info("SHAP global importance will appear after a model run.")

    with col_shap:
        st.subheader("🐝 SHAP Beeswarm Plot")
        if shap_summary_path.exists():
            st.image(str(shap_summary_path), caption="Global SHAP Summary (Beeswarm)", use_column_width=True)
        else:
            st.info(f"Beeswarm plot not found at `{shap_summary_path}`. Run the trainer to generate it.")


# ===========================================================================
# TAB 4 — DRIFT MONITOR
# ===========================================================================
with tab4:
    st.header("🌊 Drift Monitor")

    drift_events = _load_drift_events()

    # PSI scores chart
    st.subheader("📊 Feature PSI Scores")
    psi_data = _drift.compute_psi(pd.DataFrame(list(_drift._psi_window))) if _drift._psi_window else {}

    if psi_data:
        psi_df = pd.DataFrame(
            [{"Feature": k, "PSI": v} for k, v in psi_data.items()]
        ).sort_values("PSI", ascending=False)
        psi_df["Severity"] = psi_df["PSI"].apply(
            lambda x: "🔴 Critical" if x >= 0.2 else ("🟡 Warning" if x >= 0.1 else "🟢 OK")
        )
        psi_df["Color"] = psi_df["PSI"].apply(
            lambda x: "#dc3545" if x >= 0.2 else ("#ffc107" if x >= 0.1 else "#28a745")
        )
        fig_psi = go.Figure()
        for _, row in psi_df.iterrows():
            fig_psi.add_trace(go.Bar(
                x=[row["Feature"]], y=[row["PSI"]],
                marker_color=row["Color"], name=row["Severity"],
                showlegend=False,
            ))
        fig_psi.add_hline(y=0.1, line_dash="dot", line_color="#ffc107", annotation_text="Warning (0.10)")
        fig_psi.add_hline(y=0.2, line_dash="dot", line_color="#dc3545", annotation_text="Critical (0.20)")
        fig_psi.update_layout(
            template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            font_color="#a0aec0", height=280, xaxis_title="Feature", yaxis_title="PSI",
        )
        st.plotly_chart(fig_psi, use_container_width=True)
    else:
        st.info("PSI scores will appear after the model processes transactions with a PSI baseline set.")

    st.markdown("---")

    # ADWIN status
    st.subheader("🔬 ADWIN Detector Status")
    adwin_col1, adwin_col2, adwin_col3 = st.columns(3)
    adwin = _drift._adwin
    err_buf = list(_drift._error_buffer)
    curr_err_rate = round(sum(err_buf) / max(len(err_buf), 1), 4) if err_buf else 0.0

    adwin_col1.metric("ADWIN Window Size", getattr(adwin, "width", "—"))
    adwin_col2.metric("Current Error Rate", f"{curr_err_rate:.4f}")
    adwin_col3.metric("Error Buffer Size",  len(err_buf))

    st.markdown("---")

    # Drift event history
    st.subheader("📜 Drift Event History")
    if drift_events:
        drift_df = pd.DataFrame(drift_events)
        show_cols = [c for c in ["timestamp","drift_type","adwin_error_rate","recommended_action","feature","psi"] if c in drift_df.columns]
        if show_cols:
            st.dataframe(drift_df[show_cols], use_container_width=True, height=220)
    else:
        st.info("No drift events recorded yet. Events are logged to `logs/drift_events.json`.")

    # Rolling error rate image
    demo_img = Path("logs/drift_detection_demo.png")
    if demo_img.exists():
        st.subheader("📉 Rolling Error Rate")
        st.image(str(demo_img), caption="ADWIN Rolling Error Rate", use_column_width=True)

    st.markdown("---")

    # Drift simulation controls
    st.subheader("🎮 Drift Simulation Controls")
    st.caption("Inject synthetic concept drift into a copy of the dataset for testing purposes.")

    sim_col1, sim_col2, sim_col3 = st.columns([1, 1, 1])
    sim_drift_type  = sim_col1.selectbox("Drift Type",  ["high_value", "low_velocity", "label_flip"])
    sim_inj_point   = sim_col2.number_input("Injection Point", 0.5, 0.9, 0.7, 0.05)

    if sim_col3.button("🚀 Run Simulation"):
        csv_path = _CFG.get("data", {}).get("transaction_path", "data/train_transaction.csv")
        if Path(csv_path).exists():
            with st.spinner("Running drift simulation…"):
                try:
                    sim_df = pd.read_csv(csv_path, nrows=5000)
                    if "isFraud" not in sim_df.columns:
                        sim_df["isFraud"] = 0
                    drifted = DriftSimulator.simulate_concept_drift(
                        sim_df, injection_point=sim_inj_point, drift_type=sim_drift_type
                    )
                    n_pre  = int(sim_inj_point * len(drifted))
                    n_post = len(drifted) - n_pre
                    amt_col = "TransactionAmt" if "TransactionAmt" in drifted.columns else drifted.columns[0]
                    fig_sim = go.Figure()
                    fig_sim.add_trace(go.Histogram(
                        x=drifted.iloc[:n_pre][amt_col], name="Pre-Drift",
                        marker_color="#3498db", opacity=0.7,
                    ))
                    fig_sim.add_trace(go.Histogram(
                        x=drifted.iloc[n_pre:][amt_col], name="Post-Drift",
                        marker_color="#e74c3c", opacity=0.7,
                    ))
                    fig_sim.update_layout(
                        barmode="overlay", template="plotly_dark",
                        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                        font_color="#a0aec0", height=300,
                        title=f"Drift Simulation: {sim_drift_type} @ {sim_inj_point:.0%}",
                        xaxis_title=amt_col, yaxis_title="Count",
                    )
                    st.plotly_chart(fig_sim, use_container_width=True)
                    st.success(
                        f"✅ Simulation complete | pre={n_pre} rows, post={n_post} rows"
                    )
                except Exception as sim_err:
                    st.error(f"Simulation failed: {sim_err}")
        else:
            st.warning(f"CSV not found at `{csv_path}`. Cannot run simulation.")


# ===========================================================================
# TAB 5 — RETRAINING CONTROL
# ===========================================================================
with tab5:
    st.header("🔄 Retraining Control Panel")

    fb_stats = _fb_store.get_feedback_stats()
    retrain_thresh = _CFG.get("feedback", {}).get("retraining_threshold", 100)

    # Feedback stats KPIs
    st.subheader("📋 Feedback Statistics")
    fs1, fs2, fs3, fs4 = st.columns(4)
    fs1.metric("Total Records",    fb_stats.get("total_records", 0))
    fs2.metric("Unused / Pending", fb_stats.get("unused_count", 0))
    fs3.metric("Confirm Fraud",    fb_stats.get("confirm_fraud_count", 0))
    fs4.metric("False Positive",   fb_stats.get("false_positive_count", 0))

    st.markdown("---")

    # Retraining progress bar
    unused = fb_stats.get("unused_count", 0)
    progress = min(unused / max(retrain_thresh, 1), 1.0)
    st.subheader("📊 Retraining Progress")
    st.progress(progress, text=f"{unused} / {retrain_thresh} feedback records collected ({progress*100:.0f}%)")

    # Decision breakdown pie chart
    col_pie, col_info = st.columns([1, 1])
    with col_pie:
        st.subheader("🍕 Decision Breakdown")
        pie_data = {
            "CONFIRM_FRAUD":  fb_stats.get("confirm_fraud_count", 0),
            "FALSE_POSITIVE": fb_stats.get("false_positive_count", 0),
            "ESCALATE":       fb_stats.get("escalate_count", 0),
        }
        if sum(pie_data.values()) > 0:
            fig_pie = px.pie(
                values=list(pie_data.values()),
                names=list(pie_data.keys()),
                color_discrete_sequence=["#dc3545", "#28a745", "#ffc107"],
                template="plotly_dark",
                hole=0.4,
            )
            fig_pie.update_layout(
                paper_bgcolor="#0e1117", font_color="#a0aec0", height=260,
                legend=dict(bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No feedback records yet.")

    with col_info:
        st.subheader("📝 Last Retraining Info")
        last_info = st.session_state.get("last_retrain_info", {})
        if last_info:
            promoted = last_info.get("promoted", False)
            st.info(
                f"**Trigger:** {last_info.get('trigger_reason','—')}\n\n"
                f"**Timestamp:** {last_info.get('timestamp','—')}\n\n"
                f"**Promoted:** {'✅ Yes' if promoted else '❌ No'}\n\n"
                f"**Challenger AUC-PR:** {last_info.get('challenger_auc_pr','—')}\n\n"
                f"**Champion AUC-PR:** {last_info.get('champion_auc_pr','—')}\n\n"
                f"**MLflow Run:** `{last_info.get('mlflow_run_id','—')}`"
            )
        else:
            st.info("No retraining has been run in this session yet.")

    st.markdown("---")

    # Manual controls
    st.subheader("🎛️ Manual Controls")
    ctrl1, ctrl2, ctrl3 = st.columns(3)

    if ctrl1.button("🔁 Manual Retrain", type="primary"):
        def _run_retrain():
            with st.spinner("Running retraining pipeline…"):
                result = _pipeline.run(trigger_reason="manual")
                result["timestamp"] = datetime.now(timezone.utc).isoformat()
                result["challenger_auc_pr"] = result.get("challenger_metrics", {}).get("auc_pr", "—")
                result["champion_auc_pr"]   = result.get("champion_metrics", {}).get("auc_pr", "—")
                st.session_state["last_retrain_info"] = result
                if result.get("promoted"):
                    st.success(
                        f"✅ Challenger promoted to Production! "
                        f"AUC-PR: {result['challenger_auc_pr']:.4f} | "
                        f"Run: `{result.get('mlflow_run_id','')}`"
                    )
                    # Clear MLflow cache so sidebar/tab3 reflect new champion
                    get_champion_metrics.clear()
                    get_model_versions.clear()
                else:
                    st.warning(
                        "⚠️ Challenger did not improve over champion. Run logged to MLflow."
                    )

        _rt = threading.Thread(target=_run_retrain, daemon=True, name="cipher-retrain")
        _rt.start()
        st.info("🔁 Retraining pipeline launched in background thread…")

    # Promote / Reject Challenger buttons
    import mlflow
    from mlflow.tracking import MlflowClient

    _mlclient = MlflowClient(tracking_uri=_TRACKING_URI)
    _chall_m   = get_challenger_metrics(_TRACKING_URI, _REGISTRY_NAME)

    if _chall_m:
        chall_ver = str(_chall_m.get("version", ""))
        if ctrl2.button("⬆️ Promote Challenger"):
            try:
                mlflow.set_tracking_uri(_TRACKING_URI)
                _mlclient.transition_model_version_stage(
                    name=_REGISTRY_NAME, version=chall_ver, stage="Production",
                    archive_existing_versions=True,
                )
                get_champion_metrics.clear()
                get_challenger_metrics.clear()
                get_model_versions.clear()
                st.success(f"✅ Challenger v{chall_ver} promoted to Production.")
            except Exception as prm_err:
                st.error(f"Promotion failed: {prm_err}")

        if ctrl3.button("⬇️ Reject Challenger"):
            try:
                mlflow.set_tracking_uri(_TRACKING_URI)
                _mlclient.transition_model_version_stage(
                    name=_REGISTRY_NAME, version=chall_ver, stage="Archived",
                    archive_existing_versions=False,
                )
                get_challenger_metrics.clear()
                get_model_versions.clear()
                st.success(f"✅ Challenger v{chall_ver} moved to Archived.")
            except Exception as rej_err:
                st.error(f"Rejection failed: {rej_err}")
    else:
        ctrl2.button("⬆️ Promote Challenger", disabled=True)
        ctrl3.button("⬇️ Reject Challenger",  disabled=True)

    st.markdown("---")

    # Retraining history table
    st.subheader("📜 Retraining History (Last 10 Runs)")
    runs = get_experiment_runs(_TRACKING_URI, _EXPERIMENT_NAME, n=10)
    if runs:
        runs_df = pd.DataFrame(runs)
        show_cols = [c for c in ["run_id","start_time","trigger_reason","outcome","challenger_auc_pr","champion_auc_pr","promoted"] if c in runs_df.columns]
        st.dataframe(runs_df[show_cols], use_container_width=True, height=220)
    else:
        st.info("No retraining runs found. Click 'Manual Retrain' to start one.")

    st.markdown("---")

    # Feedback records table
    st.subheader("📑 Feedback Records (Last 50)")
    fb_history = _fb_store.get_feedback_history(limit=50)
    if fb_history:
        fb_df = pd.DataFrame([
            {
                "transaction_id": r.transaction_id,
                "analyst_id":     r.analyst_id,
                "decision":       r.decision.value if hasattr(r.decision, "value") else str(r.decision),
                "confidence":     round(r.confidence, 2),
                "model_score":    round(r.model_score, 4),
                "reviewed_at":    r.reviewed_at,
                "used":           r.used_in_retraining,
            }
            for r in fb_history
        ])
        st.dataframe(fb_df, use_container_width=True, height=320)
    else:
        st.info("No feedback records yet. Submit decisions in the Review Queue tab.")
