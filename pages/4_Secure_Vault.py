"""
Secure Vault Page
Session history from the PostgreSQL database via FastAPI backend.
"""
import os
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Secure Vault", page_icon="🔐", layout="wide")

API_URL = st.session_state.get("api_url", os.environ.get("BACKEND_URL", "http://localhost:8000"))

st.title("🔐 SECURE VAULT")
st.caption("Encrypted session history · Anonymization archive · Audit log — powered by PostgreSQL")

# ─── Stats ────────────────────────────────────────────────────────────────────
try:
    stats_resp = requests.get(f"{API_URL}/api/sessions/stats", timeout=5)
    stats = stats_resp.json() if stats_resp.ok else {}
except Exception:
    stats = {}

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Sessions", stats.get("total_sessions", "—"))
c2.metric("Completed", stats.get("completed", "—"))
c3.metric("CSV Processed", stats.get("csv_processed", "—"))
c4.metric("Images Processed", stats.get("image_processed", "—"))
c5.metric("Avg Risk Score", f"{stats.get('average_risk_score', 0):.1f}%" if stats else "—")

st.divider()

# ─── Session Log ──────────────────────────────────────────────────────────────
st.subheader("Anonymization Session Log")

col_refresh, col_spacer = st.columns([1, 5])
with col_refresh:
    refresh = st.button("Refresh", use_container_width=True)

try:
    sessions_resp = requests.get(f"{API_URL}/api/sessions", timeout=5)
    sessions = sessions_resp.json() if sessions_resp.ok else []
except Exception as e:
    st.error(f"Could not load sessions: {e}")
    sessions = []

if not sessions:
    st.info("No sessions found. Process a file in the Command Center to create records.")
    st.stop()

for i, s in enumerate(sessions):
    risk = s.get("risk_score")
    file_type = s.get("file_type", "csv").upper()
    created = s.get("created_at", "")[:19].replace("T", " ") if s.get("created_at") else ""

    type_colors = {"CSV": "#06b6d4", "IMAGE": "#a855f7", "AUDIO": "#f59e0b"}
    type_color = type_colors.get(file_type, "#6b7280")

    if risk is not None:
        risk_color = "#22c55e" if risk < 30 else "#f59e0b" if risk < 60 else "#ef4444"
        risk_label = "LOW RISK" if risk < 30 else "MODERATE" if risk < 60 else "HIGH"
    else:
        risk_color = "#6b7280"
        risk_label = "N/A"

    with st.container():
        left, mid, right = st.columns([3, 4, 2])
        with left:
            st.markdown(
                f'<span style="background:{type_color}20;color:{type_color};'
                f'border:1px solid {type_color}40;border-radius:4px;'
                f'padding:1px 6px;font-size:11px;font-family:monospace;">{file_type}</span> '
                f'**{s["file_name"]}**',
                unsafe_allow_html=True,
            )
            st.caption(
                f"#{s['id']:03d} · {created} · "
                f"ε={s['privacy_budget']} · k={s['k_anonymity_k']}"
                + (f" · {s['synthetic_row_count']:,} synthetic rows" if s.get("synthetic_row_count") else "")
            )
        with mid:
            status_color = "#22c55e" if s["status"] == "completed" else "#f59e0b"
            st.markdown(
                f'<span style="color:{status_color};font-size:11px;font-family:monospace;">'
                f'● {s["status"].upper()}</span>',
                unsafe_allow_html=True,
            )
        with right:
            if risk is not None:
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk,
                    number={"suffix": "%", "font": {"size": 14}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 0, "showticklabels": False},
                        "bar": {"color": risk_color, "thickness": 0.3},
                        "bgcolor": "rgba(0,0,0,0)",
                        "steps": [
                            {"range": [0, 30], "color": "rgba(34,197,94,0.1)"},
                            {"range": [30, 60], "color": "rgba(245,158,11,0.1)"},
                            {"range": [60, 100], "color": "rgba(239,68,68,0.1)"},
                        ],
                    },
                    domain={"x": [0, 1], "y": [0, 1]},
                ))
                gauge.update_layout(
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=80, paper_bgcolor="rgba(0,0,0,0)",
                    font={"color": risk_color},
                )
                st.plotly_chart(gauge, use_container_width=True, key=f"gauge_{s['id']}")
                st.caption(f"<div style='text-align:center;color:{risk_color};font-size:10px;'>{risk_label}</div>",
                           unsafe_allow_html=True)

            delete_col = st.columns(1)[0]
            if delete_col.button("Delete", key=f"del_{s['id']}", type="secondary"):
                try:
                    del_resp = requests.delete(f"{API_URL}/api/sessions/{s['id']}", timeout=5)
                    if del_resp.ok:
                        st.success(f"Session #{s['id']} deleted.")
                        st.rerun()
                    else:
                        st.error("Delete failed.")
                except Exception as e:
                    st.error(f"Error: {e}")

        st.divider()

# ─── Charts ───────────────────────────────────────────────────────────────────
if len(sessions) > 1:
    st.subheader("Session Analytics")
    c1, c2 = st.columns(2)

    with c1:
        type_counts = pd.Series([s["file_type"] for s in sessions]).value_counts()
        pie_fig = px.pie(
            values=type_counts.values, names=type_counts.index,
            title="Sessions by File Type",
            color_discrete_sequence=["#06b6d4", "#a855f7", "#f59e0b"],
            template="plotly_dark",
        )
        st.plotly_chart(pie_fig, use_container_width=True)

    with c2:
        risk_scores = [s["risk_score"] for s in sessions if s.get("risk_score") is not None]
        if risk_scores:
            risk_fig = go.Figure(go.Histogram(
                x=risk_scores, nbinsx=10,
                marker_color="#06b6d4", opacity=0.8, name="Risk Score",
            ))
            risk_fig.update_layout(
                title="Risk Score Distribution Across Sessions",
                xaxis_title="Risk Score (%)", yaxis_title="Sessions",
                template="plotly_dark", height=300,
            )
            st.plotly_chart(risk_fig, use_container_width=True)
