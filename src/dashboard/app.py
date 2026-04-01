"""Streamlit monitoring dashboard for the RAG pipeline.

Shows: documents by freshness status, retrieval quality over time,
re-embedding lag, and stale content serving log.
"""

import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import and_, create_engine, func, text
from sqlalchemy.orm import sessionmaker

from config.settings import settings
from src.db.models import (
    BronzeDocument, DeadLetterDocument, DocumentVersion, EvaluationResult,
    GoldChunk, PipelineRun, RetrievalLog, SilverDocument,
)
from src.freshness.tracker import FreshnessTracker

# ──────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Pipeline Monitor",
    page_icon="📊",
    layout="wide",
)

@st.cache_resource
def get_session():
    engine = create_engine(settings.database_url)
    return sessionmaker(bind=engine)()


db = get_session()

st.title("RAG Pipeline Monitoring Dashboard")
st.markdown("---")

# ──────────────────────────────────────────────
# Top-level KPIs
# ──────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

total_bronze = db.query(BronzeDocument).count()
total_silver = db.query(SilverDocument).count()
total_gold = db.query(GoldChunk).count()
current_gold = db.query(GoldChunk).filter(GoldChunk.is_current == True).count()
stale_gold = total_gold - current_gold
dead_letters = db.query(DeadLetterDocument).filter(DeadLetterDocument.resolved == False).count()

col1.metric("Bronze Docs", total_bronze)
col2.metric("Silver Docs", total_silver)
col3.metric("Gold Chunks (Current)", current_gold)
col4.metric("Stale Chunks", stale_gold, delta=f"-{stale_gold}" if stale_gold else "0",
            delta_color="inverse")
col5.metric("Dead Letters", dead_letters, delta_color="inverse")

# ──────────────────────────────────────────────
# Document Freshness Status
# ──────────────────────────────────────────────
st.header("Document Freshness")
col_fresh1, col_fresh2 = st.columns(2)

with col_fresh1:
    freshness_data = {
        "Status": ["Current", "Stale", "Flagged (Dead Letter)"],
        "Count": [current_gold, stale_gold, dead_letters],
    }
    fig_fresh = px.pie(
        pd.DataFrame(freshness_data), names="Status", values="Count",
        title="Chunks by Freshness Status",
        color_discrete_sequence=["#2ecc71", "#e74c3c", "#f39c12"],
    )
    st.plotly_chart(fig_fresh, use_container_width=True)

with col_fresh2:
    tracker = FreshnessTracker()
    stats = tracker.get_freshness_stats(db)
    st.metric("Freshness Ratio", f"{stats['freshness_ratio']:.1%}")
    st.metric("Avg Re-embedding Lag", f"{stats['avg_reembedding_lag_seconds']:.0f}s")
    st.metric("Multi-version Documents", stats["multi_version_documents"])
    st.metric("Total Tracked Documents", stats["total_tracked_documents"])

# ──────────────────────────────────────────────
# Pipeline Run History
# ──────────────────────────────────────────────
st.header("Pipeline Run History")

runs = db.query(PipelineRun).order_by(PipelineRun.started_at.desc()).limit(50).all()
if runs:
    run_data = []
    for r in runs:
        duration = None
        if r.completed_at and r.started_at:
            duration = (r.completed_at - r.started_at).total_seconds()
        run_data.append({
            "ID": r.id, "Type": r.run_type, "Status": r.status,
            "Started": r.started_at, "Duration (s)": duration,
            "Processed": r.documents_processed, "Failed": r.documents_failed,
            "Stale Detected": r.stale_detected, "Re-embedded": r.re_embedded,
        })
    df_runs = pd.DataFrame(run_data)
    st.dataframe(df_runs, use_container_width=True, hide_index=True)

    # Pipeline duration over time
    df_with_duration = df_runs.dropna(subset=["Duration (s)"])
    if not df_with_duration.empty:
        fig_duration = px.line(
            df_with_duration, x="Started", y="Duration (s)",
            color="Type", title="Pipeline Run Duration Over Time",
        )
        st.plotly_chart(fig_duration, use_container_width=True)
else:
    st.info("No pipeline runs yet.")

# ──────────────────────────────────────────────
# Retrieval Quality Over Time
# ──────────────────────────────────────────────
st.header("Retrieval Quality")

logs = db.query(RetrievalLog).order_by(RetrievalLog.timestamp.desc()).limit(200).all()
if logs:
    col_ret1, col_ret2 = st.columns(2)

    with col_ret1:
        log_data = [{
            "Timestamp": l.timestamp, "Latency (ms)": l.latency_ms,
            "Method": l.retrieval_method, "Served Stale": l.served_stale,
        } for l in logs if l.latency_ms]
        if log_data:
            df_logs = pd.DataFrame(log_data)
            fig_latency = px.scatter(
                df_logs, x="Timestamp", y="Latency (ms)", color="Method",
                title="Retrieval Latency Over Time",
            )
            st.plotly_chart(fig_latency, use_container_width=True)

    with col_ret2:
        stale_count = sum(1 for l in logs if l.served_stale)
        fresh_count = len(logs) - stale_count
        fig_stale = px.pie(
            pd.DataFrame({"Status": ["Fresh", "Stale"], "Count": [fresh_count, stale_count]}),
            names="Status", values="Count",
            title="Retrieval Freshness (Recent Queries)",
            color_discrete_sequence=["#2ecc71", "#e74c3c"],
        )
        st.plotly_chart(fig_stale, use_container_width=True)

    # Stale content serving log
    stale_logs = [l for l in logs if l.served_stale]
    if stale_logs:
        st.subheader("Stale Content Serving Log")
        st.warning(f"{len(stale_logs)} queries served stale content!")
        stale_data = [{
            "Time": l.timestamp, "Query": l.query[:100],
            "Latency": l.latency_ms,
        } for l in stale_logs]
        st.dataframe(pd.DataFrame(stale_data), use_container_width=True, hide_index=True)
else:
    st.info("No retrieval logs yet.")

# ──────────────────────────────────────────────
# Evaluation Results
# ──────────────────────────────────────────────
st.header("Evaluation Results")

evals = db.query(EvaluationResult).order_by(EvaluationResult.timestamp.desc()).limit(200).all()
if evals:
    # Group by eval run
    run_ids = list(set(e.eval_run_id for e in evals))
    selected_run = st.selectbox("Select Eval Run", run_ids)
    run_evals = [e for e in evals if e.eval_run_id == selected_run]

    col_eval1, col_eval2, col_eval3 = st.columns(3)
    avg_prec = sum(e.retrieval_precision or 0 for e in run_evals) / len(run_evals)
    avg_faith = sum(e.answer_faithfulness or 0 for e in run_evals) / len(run_evals)
    avg_fresh = sum(e.freshness_accuracy or 0 for e in run_evals) / len(run_evals)

    col_eval1.metric("Avg Retrieval Precision", f"{avg_prec:.2%}")
    col_eval2.metric("Avg Answer Faithfulness", f"{avg_faith:.2%}")
    col_eval3.metric("Avg Freshness Accuracy", f"{avg_fresh:.2%}")

    # Per-question breakdown
    eval_data = [{
        "Question": e.question[:80], "Precision": e.retrieval_precision,
        "Faithfulness": e.answer_faithfulness, "Freshness": e.freshness_accuracy,
    } for e in run_evals]
    st.dataframe(pd.DataFrame(eval_data), use_container_width=True, hide_index=True)

    # Radar chart for latest eval
    fig_radar = go.Figure(data=go.Scatterpolar(
        r=[avg_prec, avg_faith, avg_fresh, avg_prec],
        theta=["Retrieval Precision", "Answer Faithfulness", "Freshness Accuracy",
               "Retrieval Precision"],
        fill="toself", name=f"Run {selected_run}",
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Quality Dimensions",
    )
    st.plotly_chart(fig_radar, use_container_width=True)
else:
    st.info("No evaluation results yet. Run an evaluation to see results here.")

# ──────────────────────────────────────────────
# Dead Letter Queue
# ──────────────────────────────────────────────
st.header("Dead Letter Queue")

dead = db.query(DeadLetterDocument).filter(
    DeadLetterDocument.resolved == False
).order_by(DeadLetterDocument.error_timestamp.desc()).limit(20).all()

if dead:
    dead_data = [{
        "ID": d.id, "Source": d.source_url or "N/A", "Stage": d.stage,
        "Error": d.error_message[:100], "Time": d.error_timestamp,
        "Retries": d.retry_count,
    } for d in dead]
    st.dataframe(pd.DataFrame(dead_data), use_container_width=True, hide_index=True)
else:
    st.success("No unresolved dead letters.")

# ──────────────────────────────────────────────
# Reconciliation Check
# ──────────────────────────────────────────────
st.header("Vector Store Reconciliation")

if st.button("Run Reconciliation Check"):
    with st.spinner("Running reconciliation..."):
        recon = tracker.reconcile_vector_store(db)

    if recon["is_healthy"]:
        st.success("Vector store is healthy - no drift detected.")
    else:
        st.error("Issues detected!")

    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    col_r1.metric("Total Current Chunks", recon["total_current_chunks"])
    col_r2.metric("Stale Chunks Found", recon["stale_chunks_found"])
    col_r3.metric("Orphaned Chunks", recon["orphaned_chunks"])
    col_r4.metric("Missing Embeddings", recon["missing_embeddings"])

    if recon["issues"]:
        st.subheader("Issues")
        for issue in recon["issues"][:20]:
            st.text(f"  - {issue}")
