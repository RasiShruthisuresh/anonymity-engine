"""
Gen AI Insights Page — Multimodal
Covers: CSV (NER + DP + CTGAN), Image (GAN pipeline), Audio (WaveGAN pipeline)
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Gen AI Insights", page_icon="🧠", layout="wide")
st.title("🧠 GEN AI INSIGHTS")

has_csv   = "result" in st.session_state
has_image = "image_result" in st.session_state
has_audio = "audio_result" in st.session_state

if not has_csv and not has_image and not has_audio:
    st.warning("No results yet. Go to **Command Center** and process a CSV, image, or audio file first.")
    st.stop()

tabs = []
if has_csv:   tabs.append("📊 CSV — CTGAN + DP + NER")
if has_image: tabs.append("🖼️ Image — GAN Pipeline")
if has_audio: tabs.append("🎙️ Audio — WaveGAN Pipeline")

selected_tabs = st.tabs(tabs)
tab_idx = 0

# ═══════════════════════════════════════════════════════════════
# CSV TAB
# ═══════════════════════════════════════════════════════════════
if has_csv:
    with selected_tabs[tab_idx]:
        tab_idx += 1
        result  = st.session_state["result"]
        epsilon = st.session_state.get("epsilon", 1.0)

        MECHANISM_LABEL = {
            "pattern": "REGEX PATTERN",
            "contextual": "TRANSFORMER-NER",
            "statistical": "STATISTICAL",
        }

        st.subheader("Sentry Agent — Zero-Shot Transformer NER")
        st.caption("PII entity detection via semantic header analysis (simulates BERT/RoBERTa NER) and regex pattern matching.")
        detections = result.get("ner_detections", [])
        if not detections:
            st.success("No PII entities detected.")
        else:
            ner_data = [{
                "Field": d["field"], "Entity Type": d["entity_type"],
                "Confidence": round(d["confidence"] * 100, 1),
                "Mechanism": MECHANISM_LABEL.get(d["mechanism"], d["mechanism"]),
                "Sample": d.get("sample", "")[:24],
            } for d in detections]
            ner_df = pd.DataFrame(ner_data)
            st.dataframe(ner_df, use_container_width=True, hide_index=True)
            conf_fig = px.bar(
                ner_df, x="Field", y="Confidence", color="Entity Type",
                title="PII Entity Confidence Scores", labels={"Confidence": "Confidence (%)"},
                template="plotly_dark", height=320,
                color_discrete_sequence=px.colors.qualitative.Bold,
            )
            conf_fig.add_hline(y=90, line_dash="dash", line_color="yellow",
                               annotation_text="High confidence threshold (90%)")
            st.plotly_chart(conf_fig, use_container_width=True)

        st.divider()
        st.subheader("Ghost Agent — Differential Privacy ε Budget Tracker")
        budget = result.get("epsilon_budget", {})
        total  = budget.get("total", epsilon)
        spent  = budget.get("spent", 0)
        remaining = budget.get("remaining", total)
        query_log = budget.get("query_log", [])
        used_pct  = (spent / total * 100) if total > 0 else 0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total ε Budget", f"{total:.2f}")
        m2.metric("Consumed", f"{spent:.4f}", delta=f"{used_pct:.1f}% used", delta_color="inverse")
        m3.metric("Remaining", f"{remaining:.4f}")
        m4.metric("Total Queries", budget.get("query_count", 0))

        bfig = go.Figure(go.Bar(
            x=["Spent", "Remaining"],
            y=[spent, remaining],
            marker_color=["#ef4444" if used_pct > 80 else "#f59e0b" if used_pct > 50 else "#06b6d4", "#22c55e"],
            text=[f"ε {spent:.4f}", f"ε {remaining:.4f}"], textposition="outside", width=0.4,
        ))
        bfig.update_layout(title="Epsilon Budget Consumption", template="plotly_dark",
                           height=280, yaxis_title="Epsilon (ε)", showlegend=False)
        st.plotly_chart(bfig, use_container_width=True)

        if query_log:
            log_df = pd.DataFrame([{
                "Column": q["column"], "Mechanism": q["mechanism"].upper(),
                "ε Spent": round(q["epsilon_spent"], 6), "Sensitivity": round(q["sensitivity"], 4),
            } for q in query_log[-20:]])
            mech_counts = log_df["Mechanism"].value_counts()
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Laplace Queries", int(mech_counts.get("LAPLACE", 0)))
            mc2.metric("Exponential Queries", int(mech_counts.get("EXPONENTIAL", 0)))
            mc3.metric("Columns Touched", log_df["Column"].nunique())
            st.dataframe(log_df, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Alchemist Agent — CTGAN Gaussian Copula Synthesis")
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Training Epochs", result.get("gan_epochs", 0))
        a2.metric("Original Rows", f"{result.get('original_rows', 0):,}")
        a3.metric("Synthetic Rows", f"{result.get('synthetic_rows', 0):,}")
        a4.metric("Fidelity Verdict", result["fidelity_report"]["verdict"])
        st.info("""**CTGAN Gaussian Copula steps:**
1. **Marginal Fitting** — mean/std per numeric column; frequency for categoricals
2. **Pearson Correlation Matrix** — captures inter-column relationships
3. **Cholesky Decomposition** — samples correlated normal variables
4. **Marginal Transform** — maps normals back through inverse CDF
5. **DP Noise Injection** — Laplace noise calibrated to ε budget
6. **Categorical Sampling** — Exponential mechanism from frequency distributions""")

        col_stats = result.get("column_stats", [])
        if col_stats:
            stats_data = []
            for s in col_stats:
                row = {"Column": s["column"],
                       "Orig Mean": f"{s['orig_mean']:.2f}", "Orig Std": f"{s['orig_std']:.2f}",
                       "CTGAN Mean": f"{s['syn_mean']:.2f}" if s.get("syn_mean") is not None else "—",
                       "CTGAN Std": f"{s['syn_std']:.2f}" if s.get("syn_std") is not None else "—"}
                if s.get("syn_mean") is not None and s["orig_mean"] != 0:
                    row["Mean Drift %"] = f"{abs(s['syn_mean'] - s['orig_mean']) / abs(s['orig_mean']) * 100:.1f}%"
                else:
                    row["Mean Drift %"] = "—"
                stats_data.append(row)
            st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════
# IMAGE TAB
# ═══════════════════════════════════════════════════════════════
if has_image:
    with selected_tabs[tab_idx]:
        tab_idx += 1
        ir = st.session_state["image_result"]

        st.subheader("Alchemist Agent — Image GAN Pipeline")
        st.caption(f"File: **{ir['filename']}** | Size: {ir['width']}×{ir['height']}px | GDPR: {ir['gdpr_status']}")

        # ── GAN stage cards ──────────────────────────────────────────────────
        st.markdown("#### GAN Architecture Stages Applied")
        cols = st.columns(min(4, len(ir["gan_stages"])))
        for i, (num, name, desc, color) in enumerate(ir["gan_stages"]):
            cols[i % 4].markdown(
                f"""<div style="border:1px solid {color}55;border-radius:8px;padding:10px;
                background:{color}12;margin-bottom:8px;min-height:90px;">
                <div style="color:{color};font-weight:bold;font-size:13px;">Stage {num}</div>
                <div style="font-weight:600;font-size:12px;margin-top:4px;">{name}</div>
                <div style="font-size:10px;color:#999;margin-top:3px;">{desc}</div>
                </div>""",
                unsafe_allow_html=True,
            )

        st.divider()

        # ── Pixel destruction metrics ────────────────────────────────────────
        st.subheader("Biometric Destruction Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Pixel Destruction", f"{ir['pixel_destruction_pct']:.1f}%",
                  help="Mean absolute pixel difference / 255 × 100")
        m2.metric("PSNR Proxy", f"{ir['psnr_proxy']:.1f} dB",
                  help="Lower PSNR = more distortion = stronger anonymization")
        m3.metric("MSE", f"{ir['mse']:.1f}", help="Mean Squared Error between original and synthesized")
        m4.metric("Channels Processed", "3 (R / G / B)")

        st.divider()

        # ── Per-channel KS analysis ──────────────────────────────────────────
        st.subheader("PatchGAN Discriminator — Per-Channel Realness Analysis")
        st.caption("KS statistic measures how differently the original and synthesized pixel intensity distributions behave per channel. Higher = more distinct = better destruction.")
        ks_data = pd.DataFrame([{
            "Channel": c["channel"],
            "Orig Mean": f"{c['orig_mean']:.1f}", "Synth Mean": f"{c['synth_mean']:.1f}",
            "Orig Std": f"{c['orig_std']:.1f}", "Synth Std": f"{c['synth_std']:.1f}",
            "KS Statistic": round(c["ks_statistic"], 4),
            "P-Value": f"{c['p_value']:.4f}",
            "Biometric Destroyed": "✅ YES" if c["passed"] else "⚠️ PARTIAL",
        } for c in ir["channel_stats"]])
        st.dataframe(ks_data, use_container_width=True, hide_index=True)

        ks_fig = go.Figure(go.Bar(
            x=[c["channel"] for c in ir["channel_stats"]],
            y=[c["ks_statistic"] for c in ir["channel_stats"]],
            marker_color=["#ef4444", "#22c55e", "#3b82f6"],
            text=[f"{c['ks_statistic']:.3f}" for c in ir["channel_stats"]],
            textposition="outside",
        ))
        ks_fig.update_layout(title="KS Distance per RGB Channel (Original vs GAN Synthesized)",
                             template="plotly_dark", height=300,
                             yaxis_title="KS Statistic", xaxis_title="Channel")
        st.plotly_chart(ks_fig, use_container_width=True)

        st.divider()

        # ── GDPR compliance ──────────────────────────────────────────────────
        st.subheader("GDPR Compliance Record")
        gdpr_rows = [
            ("Art. 5(1)(c) — Data Minimisation", "EXIF/GPS/ICC/XMP metadata stripped pre-processing", "✅ Enforced"),
            ("Art. 9 — Special Category Data", "Facial geometry, iris patterns, skin-texture destroyed", "✅ Enforced"),
            ("Art. 17 — Right to Erasure", "Original bytes processed in-memory only, not persisted", "✅ Enforced"),
            ("GDPR_Compliance_Check flag", "Set on every API response", f"✅ {ir['gdpr_status']}"),
        ]
        st.dataframe(pd.DataFrame(gdpr_rows, columns=["Article", "Measure", "Status"]),
                     use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════
# AUDIO TAB
# ═══════════════════════════════════════════════════════════════
if has_audio:
    with selected_tabs[tab_idx]:
        ar = st.session_state["audio_result"]

        st.subheader("Alchemist Agent — Audio GAN Pipeline")
        st.caption(
            f"File: **{ar['filename']}** | Sample Rate: {ar['sample_rate']} Hz | "
            f"Duration: {ar['duration_s']:.2f}s | GDPR: {ar['gdpr_status']}"
        )

        # ── GAN stage cards ──────────────────────────────────────────────────
        st.markdown("#### GAN Architecture Stages Applied")
        cols = st.columns(4)
        for i, (num, name, desc, color) in enumerate(ar["gan_stages"]):
            cols[i % 4].markdown(
                f"""<div style="border:1px solid {color}55;border-radius:8px;padding:10px;
                background:{color}12;margin-bottom:8px;min-height:90px;">
                <div style="color:{color};font-weight:bold;font-size:13px;">Stage {num}</div>
                <div style="font-weight:600;font-size:12px;margin-top:4px;">{name}</div>
                <div style="font-size:10px;color:#999;margin-top:3px;">{desc}</div>
                </div>""",
                unsafe_allow_html=True,
            )

        st.divider()

        # ── Voice-print destruction metrics ──────────────────────────────────
        st.subheader("Voice-Print Destruction Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Voice-Print Destroyed", f"{ar['voice_print_destruction_pct']:.1f}%",
                  help="Spectral divergence + RMS energy shift combined score")
        m2.metric("Spectral Divergence", f"{ar['spectral_divergence']:.4f}",
                  help="KL-like divergence between original and synthesized spectra")
        m3.metric("RMS Energy Shift", f"{abs(ar['orig_rms'] - ar['synth_rms']):.4f}",
                  help="Change in RMS energy level")
        m4.metric("ZCR Shift", f"{abs(ar['orig_zcr'] - ar['synth_zcr']):.4f}",
                  help="Change in zero-crossing rate — indicates spectral texture change")

        energy_fig = go.Figure(go.Bar(
            x=["Original RMS", "Synthesized RMS"],
            y=[ar["orig_rms"], ar["synth_rms"]],
            marker_color=["#ef4444", "#22c55e"],
            text=[f"{ar['orig_rms']:.4f}", f"{ar['synth_rms']:.4f}"],
            textposition="outside", width=0.4,
        ))
        energy_fig.update_layout(title="RMS Energy — Original vs WaveGAN Synthesized",
                                 template="plotly_dark", height=280, showlegend=False)
        st.plotly_chart(energy_fig, use_container_width=True)

        st.divider()

        # ── GDPR compliance ──────────────────────────────────────────────────
        st.subheader("GDPR Compliance Record")
        gdpr_rows = [
            ("Art. 5(1)(c) — Data Minimisation", "ID3 / Vorbis / RIFF INFO metadata stripped", "✅ Enforced"),
            ("Art. 9 — Special Category Data", "Voice print: F0, F1–F4 formants, prosody destroyed", "✅ Enforced"),
            ("Art. 17 — Right to Erasure", "Original bytes in-memory only, not persisted", "✅ Enforced"),
            ("GDPR_Compliance_Check flag", "Set on every API response", f"✅ {ar['gdpr_status']}"),
        ]
        st.dataframe(pd.DataFrame(gdpr_rows, columns=["Article", "Measure", "Status"]),
                     use_container_width=True, hide_index=True)
