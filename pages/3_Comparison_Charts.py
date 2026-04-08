"""
Comparison Charts Page — Multimodal
Covers: CSV (distribution histograms), Image (RGB channel histograms), Audio (waveform + spectrum)
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Comparison Charts", page_icon="📊", layout="wide")
st.title("📊 COMPARISON CHARTS")

has_csv   = "result" in st.session_state
has_image = "image_result" in st.session_state
has_audio = "audio_result" in st.session_state

if not has_csv and not has_image and not has_audio:
    st.warning("No results yet. Go to **Command Center** and process a CSV, image, or audio file first.")
    st.stop()

tabs = []
if has_csv:   tabs.append("📊 CSV — Distribution Comparison")
if has_image: tabs.append("🖼️ Image — RGB Channel Analysis")
if has_audio: tabs.append("🎙️ Audio — Waveform & Spectrum")

selected_tabs = st.tabs(tabs)
tab_idx = 0

# ═══════════════════════════════════════════════════════════════
# CSV TAB
# ═══════════════════════════════════════════════════════════════
if has_csv:
    with selected_tabs[tab_idx]:
        tab_idx += 1
        result    = st.session_state["result"]
        col_stats = result.get("column_stats", [])

        st.caption("Distribution comparison: Original vs Anonymized (K-Anon+DP) vs Synthetic (CTGAN Gaussian Copula)")

        if not col_stats:
            st.info("No numeric columns found for chart comparison.")
        else:
            col_names    = [s["column"] for s in col_stats]
            selected_col = st.selectbox("Select numeric column", col_names)
            selected     = next((s for s in col_stats if s["column"] == selected_col), None)

            if selected:
                fig = go.Figure()
                if selected.get("orig_sample"):
                    fig.add_trace(go.Histogram(x=selected["orig_sample"], name="Original",
                                               opacity=0.65, marker_color="#ef4444", nbinsx=40))
                if selected.get("syn_sample"):
                    fig.add_trace(go.Histogram(x=selected["syn_sample"], name="Synthetic (CTGAN)",
                                               opacity=0.65, marker_color="#22c55e", nbinsx=40))
                fig.update_layout(barmode="overlay", template="plotly_dark", height=360,
                                  title=f"Distribution: {selected_col}",
                                  xaxis_title=selected_col, yaxis_title="Count",
                                  legend=dict(orientation="h", yanchor="bottom", y=1.02))
                st.plotly_chart(fig, use_container_width=True)

                m1, m2, m3 = st.columns(3)
                m1.metric("Original Mean", f"{selected['orig_mean']:.2f}")
                m2.metric("CTGAN Mean", f"{selected['syn_mean']:.2f}" if selected.get("syn_mean") is not None else "—",
                          delta=f"{(selected['syn_mean'] - selected['orig_mean']):.2f}" if selected.get("syn_mean") else None)
                m3.metric("Original Std", f"{selected['orig_std']:.2f}")

            st.divider()
            st.subheader("All Columns — Mean Comparison")
            all_data = []
            for s in col_stats:
                all_data.append({"Column": s["column"], "Metric": "Original Mean", "Value": s["orig_mean"]})
                if s.get("syn_mean") is not None:
                    all_data.append({"Column": s["column"], "Metric": "CTGAN Mean", "Value": s["syn_mean"]})
            if all_data:
                gf = px.bar(pd.DataFrame(all_data), x="Column", y="Value", color="Metric",
                            barmode="group", template="plotly_dark", height=350,
                            color_discrete_map={"Original Mean": "#ef4444", "CTGAN Mean": "#22c55e"},
                            title="Original vs CTGAN Synthetic — Mean Values per Column")
                st.plotly_chart(gf, use_container_width=True)

            st.divider()
            st.subheader("Summary Statistics Table")
            stats_rows = []
            for s in col_stats:
                row = {"Column": s["column"],
                       "Orig Mean": f"{s['orig_mean']:.3f}", "Orig Std": f"{s['orig_std']:.3f}",
                       "CTGAN Mean": f"{s['syn_mean']:.3f}" if s.get("syn_mean") is not None else "—",
                       "CTGAN Std": f"{s['syn_std']:.3f}" if s.get("syn_std") is not None else "—"}
                if s.get("syn_mean") is not None and s["orig_mean"] != 0:
                    row["Mean Drift %"] = f"{abs(s['syn_mean'] - s['orig_mean']) / (abs(s['orig_mean']) + 1e-9) * 100:.1f}%"
                else:
                    row["Mean Drift %"] = "—"
                stats_rows.append(row)
            st.dataframe(pd.DataFrame(stats_rows), use_container_width=True, hide_index=True)

            st.divider()
            st.subheader("KS Distance Heatmap")
            ks_distances = result["fidelity_report"].get("ks_distances", [])
            if ks_distances:
                ks_df = pd.DataFrame([{"Column": d["column"],
                                       "KS Distance %": round(d["ks_statistic"] * 100, 2),
                                       "P-Value": round(d["p_value"], 4),
                                       "Status": "PASS" if d["passed"] else "WARN"} for d in ks_distances])
                hf = px.bar(ks_df, x="Column", y="KS Distance %",
                            color="KS Distance %", color_continuous_scale="RdYlGn_r",
                            title="KS Distance Heatmap — how far synthetic diverges from original",
                            template="plotly_dark", height=320)
                hf.add_hline(y=10, line_dash="dash", line_color="white", annotation_text="10% Threshold")
                st.plotly_chart(hf, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# IMAGE TAB
# ═══════════════════════════════════════════════════════════════
if has_image:
    with selected_tabs[tab_idx]:
        tab_idx += 1
        ir = st.session_state["image_result"]

        st.caption(
            f"**{ir['filename']}** — {ir['width']}×{ir['height']}px | "
            "RGB channel intensity distributions: Original vs GAN Synthesized"
        )

        # ── Channel selector ─────────────────────────────────────────────────
        selected_ch = st.selectbox("Select channel", ["R", "G", "B", "All Channels"])

        ch_colors = {"R": "#ef4444", "G": "#22c55e", "B": "#3b82f6"}

        if selected_ch == "All Channels":
            for c in ir["channel_stats"]:
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=c["orig_sample"], name=f"Original ({c['channel']})",
                    opacity=0.65, marker_color=ch_colors[c["channel"]], nbinsx=60,
                ))
                fig.add_trace(go.Histogram(
                    x=c["synth_sample"], name=f"GAN Synthesized ({c['channel']})",
                    opacity=0.55, marker_color="#a855f7", nbinsx=60,
                ))
                fig.update_layout(
                    barmode="overlay", template="plotly_dark", height=300,
                    title=f"Channel {c['channel']} — Pixel Intensity Distribution",
                    xaxis_title="Pixel Intensity (0–255)", yaxis_title="Count",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            c = next(x for x in ir["channel_stats"] if x["channel"] == selected_ch)
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=c["orig_sample"], name="Original",
                opacity=0.65, marker_color=ch_colors[selected_ch], nbinsx=60,
            ))
            fig.add_trace(go.Histogram(
                x=c["synth_sample"], name="GAN Synthesized",
                opacity=0.55, marker_color="#a855f7", nbinsx=60,
            ))
            fig.update_layout(
                barmode="overlay", template="plotly_dark", height=360,
                title=f"Channel {selected_ch} — Pixel Intensity Distribution",
                xaxis_title="Pixel Intensity (0–255)", yaxis_title="Count",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig, use_container_width=True)

            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Original Mean", f"{c['orig_mean']:.1f}")
            mc2.metric("Synth Mean", f"{c['synth_mean']:.1f}",
                       delta=f"{c['synth_mean'] - c['orig_mean']:.1f}")
            mc3.metric("Original Std", f"{c['orig_std']:.1f}")
            mc4.metric("KS Distance", f"{c['ks_statistic']:.3f}")

        st.divider()

        # ── Channel mean comparison ──────────────────────────────────────────
        st.subheader("RGB Channel Mean Comparison")
        bar_data = []
        for c in ir["channel_stats"]:
            bar_data.extend([
                {"Channel": c["channel"], "Type": "Original", "Mean": c["orig_mean"]},
                {"Channel": c["channel"], "Type": "GAN Synthesized", "Mean": c["synth_mean"]},
            ])
        bf = px.bar(pd.DataFrame(bar_data), x="Channel", y="Mean", color="Type",
                    barmode="group", template="plotly_dark", height=300,
                    color_discrete_map={"Original": "#ef4444", "GAN Synthesized": "#22c55e"},
                    title="Per-Channel Mean Pixel Intensity — Original vs GAN Synthesized")
        st.plotly_chart(bf, use_container_width=True)

        st.divider()

        # ── Pixel destruction summary ────────────────────────────────────────
        st.subheader("Biometric Pixel Destruction Summary")
        st.dataframe(pd.DataFrame([{
            "Channel": c["channel"],
            "Mean Shift": f"{c['synth_mean'] - c['orig_mean']:.1f}",
            "Std Shift": f"{c['synth_std'] - c['orig_std']:.1f}",
            "KS Statistic": round(c["ks_statistic"], 4),
            "Biometric Status": "✅ Destroyed" if c["passed"] else "⚠️ Partial",
        } for c in ir["channel_stats"]]), use_container_width=True, hide_index=True)

        st.divider()

        # ── Global pixel metrics ─────────────────────────────────────────────
        st.subheader("Global Pixel Metrics")
        gm_data = {
            "Metric": ["MSE", "PSNR (dB)", "Pixel Destruction %"],
            "Value": [f"{ir['mse']:.2f}", f"{ir['psnr_proxy']:.2f}", f"{ir['pixel_destruction_pct']:.2f}%"],
            "Interpretation": [
                "Mean squared pixel error (higher = more change)",
                "Lower PSNR = more distortion = stronger anonymisation",
                "Mean absolute pixel change as % of full range",
            ],
        }
        st.dataframe(pd.DataFrame(gm_data), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════
# AUDIO TAB
# ═══════════════════════════════════════════════════════════════
if has_audio:
    with selected_tabs[tab_idx]:
        ar = st.session_state["audio_result"]

        st.caption(
            f"**{ar['filename']}** | {ar['sample_rate']} Hz | {ar['duration_s']:.2f}s — "
            "Waveform and frequency spectrum: Original vs WaveGAN Synthesized"
        )

        chart_type = st.radio("View", ["Waveform", "FFT Spectrum", "Both"], horizontal=True)

        # ── Waveform ──────────────────────────────────────────────────────────
        if chart_type in ("Waveform", "Both"):
            st.subheader("Waveform Comparison — Time Domain")
            wf = go.Figure()
            if ar.get("orig_waveform"):
                x_orig = list(range(len(ar["orig_waveform"])))
                wf.add_trace(go.Scatter(x=x_orig, y=ar["orig_waveform"], name="Original",
                                        mode="lines", line=dict(color="#ef4444", width=1), opacity=0.8))
            if ar.get("synth_waveform"):
                x_synth = list(range(len(ar["synth_waveform"])))
                wf.add_trace(go.Scatter(x=x_synth, y=ar["synth_waveform"], name="WaveGAN Synthesized",
                                        mode="lines", line=dict(color="#22c55e", width=1), opacity=0.8))
            wf.update_layout(
                title="Waveform — Original vs WaveGAN Synthesized",
                xaxis_title="Sample Index (downsampled)", yaxis_title="Amplitude",
                template="plotly_dark", height=360,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(wf, use_container_width=True)

        # ── FFT Spectrum ──────────────────────────────────────────────────────
        if chart_type in ("FFT Spectrum", "Both"):
            st.subheader("FFT Magnitude Spectrum — Frequency Domain")
            sf_fig = go.Figure()
            if ar.get("orig_spectrum") and ar.get("freqs"):
                sf_fig.add_trace(go.Scatter(
                    x=ar["freqs"], y=ar["orig_spectrum"], name="Original",
                    mode="lines", line=dict(color="#ef4444", width=1.5), opacity=0.85,
                ))
                sf_fig.add_trace(go.Scatter(
                    x=ar["freqs"], y=ar["synth_spectrum"], name="WaveGAN Synthesized",
                    mode="lines", line=dict(color="#22c55e", width=1.5), opacity=0.85,
                ))
            sf_fig.update_layout(
                title="FFT Spectrum — Original vs WaveGAN Synthesized",
                xaxis_title="Frequency (Hz)", yaxis_title="Magnitude",
                template="plotly_dark", height=360,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(sf_fig, use_container_width=True)

        st.divider()

        # ── Stats table ───────────────────────────────────────────────────────
        st.subheader("Audio Statistics Comparison")
        stats_table = pd.DataFrame([
            {"Metric": "RMS Energy", "Original": f"{ar['orig_rms']:.5f}",
             "WaveGAN Synthesized": f"{ar['synth_rms']:.5f}",
             "Change": f"{ar['synth_rms'] - ar['orig_rms']:.5f}"},
            {"Metric": "Zero-Crossing Rate", "Original": f"{ar['orig_zcr']:.5f}",
             "WaveGAN Synthesized": f"{ar['synth_zcr']:.5f}",
             "Change": f"{ar['synth_zcr'] - ar['orig_zcr']:.5f}"},
            {"Metric": "Spectral Divergence (KL)", "Original": "—",
             "WaveGAN Synthesized": f"{ar['spectral_divergence']:.5f}", "Change": "—"},
            {"Metric": "Voice-Print Destruction", "Original": "0%",
             "WaveGAN Synthesized": f"{ar['voice_print_destruction_pct']:.1f}%", "Change": "—"},
        ])
        st.dataframe(stats_table, use_container_width=True, hide_index=True)

        st.divider()

        # ── Multi-scale discriminator visualization ───────────────────────────
        st.subheader("MelGAN Multi-Scale Discriminator — Energy per Scale")
        if ar.get("orig_spectrum"):
            n = len(ar["orig_spectrum"])
            scales = {"Full (1×)": ar["orig_spectrum"],
                      "2× Downsample": ar["orig_spectrum"][::2],
                      "4× Downsample": ar["orig_spectrum"][::4]}
            disc_fig = go.Figure()
            colors = ["#ef4444", "#f59e0b", "#06b6d4"]
            for (name, data), color in zip(scales.items(), colors):
                disc_fig.add_trace(go.Scatter(
                    x=list(range(len(data))), y=data, name=name,
                    mode="lines", line=dict(color=color, width=1.2), opacity=0.8,
                ))
            disc_fig.update_layout(
                title="MelGAN Discriminator — Original Spectrum at 3 Temporal Scales",
                xaxis_title="Frequency Bin", yaxis_title="Magnitude",
                template="plotly_dark", height=320,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(disc_fig, use_container_width=True)
