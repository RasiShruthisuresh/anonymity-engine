"""
ANONYMITY ENGINE — Streamlit Command Center (Main Page)
Handles file upload, runs the full 5-agent Gen AI pipeline via FastAPI backend.
"""
import io
import os
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from scipy.stats import ks_2samp

API_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Anonymity Engine",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔒 ANONYMITY ENGINE")
    st.caption("Privacy Architecture v2.0 — Python + FastAPI")
    st.divider()

    st.subheader("⚙️ Privacy Controls")
    epsilon = st.slider(
        "Epsilon (ε) — DP Budget",
        min_value=0.1, max_value=10.0, value=1.0, step=0.1,
        help="Lower ε = stronger privacy but more distortion."
    )
    st.caption("ε < 0.5 Strong  |  ε 1–3 Balanced  |  ε > 5 Weak")

    k_value = st.slider(
        "K-Anonymity (K)",
        min_value=2, max_value=20, value=5, step=1,
        help="Each record indistinguishable from at least K-1 others."
    )

    st.divider()
    st.caption("Navigate using the sidebar pages ↓")

    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        if r.ok:
            st.success("Backend: ONLINE")
        else:
            st.error("Backend: ERROR")
    except Exception:
        st.warning("Backend: CONNECTING...")

# ── Store controls in session state ─────────────────────────────────────────
st.session_state["epsilon"] = epsilon
st.session_state["k_value"] = k_value
st.session_state["api_url"] = API_URL

# ── Header ───────────────────────────────────────────────────────────────────
st.title("COMMAND CENTER")
st.caption("Multi-agent privacy anonymization — CTGAN · Differential Privacy · Zero-Shot NER · K-Anonymity · Multimodal Synthesis · GDPR Compliant")

c1, c2, c3 = st.columns(3)
c1.metric("Privacy Budget", f"ε = {epsilon}")
c2.metric("K-Anonymity", f"K = {k_value}")
c3.metric("Mode", "Strong" if epsilon < 1 else "Balanced" if epsilon < 5 else "Weak")

st.divider()

# ── Agent Pipeline Display ───────────────────────────────────────────────────
st.subheader("5-Agent Gen AI Pipeline")
agent_cols = st.columns(5)
agents = [
    ("S",  "The Sentry",    "Zero-Shot NER · PII Detection",      "#3b82f6"),
    ("A",  "The Auditor",   "Risk Modeling · GDPR Guardrails",    "#f59e0b"),
    ("G",  "The Ghost",     "Laplace DP · Metadata Purge",        "#a855f7"),
    ("AL", "The Alchemist", "CTGAN · Image · Audio Synthesis",    "#22c55e"),
    ("J",  "The Judge",     "Fidelity · GDPR Compliance Report",  "#06b6d4"),
]
for col, (icon, name, sub, color) in zip(agent_cols, agents):
    status = "ACTIVE" if "result" in st.session_state else "STANDBY"
    col.markdown(
        f"""<div style="border:1px solid {color}40;border-radius:8px;padding:12px;
        text-align:center;background:{color}10;">
        <div style="font-size:22px;font-weight:bold;color:{color};">{icon}</div>
        <div style="font-weight:600;margin-top:4px;font-size:13px;">{name}</div>
        <div style="font-size:10px;color:#888;margin-top:2px;">{sub}</div>
        <div style="font-size:10px;color:{color};margin-top:6px;">{status}</div>
        </div>""",
        unsafe_allow_html=True,
    )

st.divider()

# ── File Upload ───────────────────────────────────────────────────────────────
st.subheader("Upload File")
st.markdown(
    """
    <div style="border:1px solid #06b6d440;border-radius:8px;padding:12px;
    background:#06b6d410;margin-bottom:12px;">
    <b style="color:#06b6d4;">Multimodal Pipeline — all three data types supported</b><br>
    <span style="font-size:13px;color:#aaa;">
    📊 <b>Tabular (.csv)</b> — Full 5-agent pipeline: NER · Differential Privacy · K-Anonymity · CTGAN synthesis<br>
    🖼️ <b>Images (.jpg .png .webp)</b> — GDPR biometric destruction: EXIF purge · Gaussian blur · Adversarial noise · Colour-jitter<br>
    🎙️ <b>Audio (.wav .mp3)</b> — Voice-print destruction: metadata strip · F0 pitch-shift · Formant warp · Temporal jitter
    </span>
    </div>
    """,
    unsafe_allow_html=True,
)
uploaded = st.file_uploader(
    "Drop a CSV, image (.jpg / .png / .webp), or audio file (.wav / .mp3) — GDPR-compliant processing guaranteed",
    type=["csv", "jpg", "jpeg", "png", "webp", "wav", "mp3"],
)

if uploaded is None:
    st.info("Upload a file above to launch the pipeline.")
    st.stop()

ext = uploaded.name.rsplit(".", 1)[-1].lower()

# ── CSV Processing ────────────────────────────────────────────────────────────
if ext == "csv":
    df = pd.read_csv(uploaded)
    st.success(f"Loaded **{uploaded.name}** — {len(df):,} rows × {len(df.columns)} columns")

    with st.expander("Preview original data"):
        st.dataframe(df.head(8), use_container_width=True)

    if st.button("🚀 Run Full Gen AI Pipeline", type="primary", use_container_width=True):
        with st.status("Running 5-agent anonymization pipeline...", expanded=True) as status_box:
            steps = [
                ("Sentry",    "Zero-Shot Transformer NER — scanning PII entities..."),
                ("Auditor",   "Risk modeling & Quasi-Identifier analysis..."),
                ("Ghost",     "Laplace + Exponential Differential Privacy..."),
                ("Alchemist", "CTGAN Gaussian Copula synthesis in progress..."),
                ("Judge",     "Kolmogorov-Smirnov fidelity test..."),
            ]
            for name, msg in steps:
                st.write(f"**Agent — The {name}:** {msg}")

            files = {"file": (uploaded.name, df.to_csv(index=False).encode(), "text/csv")}
            data = {"epsilon": epsilon, "k": k_value}
            try:
                resp = requests.post(f"{API_URL}/api/process/csv", files=files, data=data, timeout=120)
                resp.raise_for_status()
                result = resp.json()
                st.session_state["result"] = result
                st.session_state["orig_df"] = df
                status_box.update(label="Pipeline complete!", state="complete")
            except Exception as e:
                status_box.update(label="Pipeline failed.", state="error")
                st.error(f"Error: {e}")
                st.stop()

# ── IMAGE ──────────────────────────────────────────────────────────────────────
elif ext in ("jpg", "jpeg", "png", "webp"):
    image_bytes = uploaded.read()
    original_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    st.success(f"Loaded **{uploaded.name}** — {original_img.width}×{original_img.height}px")

    with st.spinner("Running GDPR biometric destruction pipeline..."):
        try:
            resp = requests.post(
                f"{API_URL}/api/process/image",
                files={"file": (uploaded.name, image_bytes, f"image/{ext}")},
                data={"sigma": 20},
                timeout=30,
            )
            resp.raise_for_status()
            synth_img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            gdpr_status = resp.headers.get("X-GDPR-Compliance-Check", "Unknown")
            gdpr_articles = resp.headers.get("X-GDPR-Articles", "")
        except Exception as e:
            st.error(f"Backend error: {e}")
            st.stop()

    if gdpr_status == "Passed":
        st.markdown(
            f"""<div style="border:1px solid #22c55e;border-radius:6px;padding:8px 14px;
            background:#22c55e15;display:inline-block;margin-bottom:8px;">
            ✅ <b style="color:#22c55e;">GDPR_Compliance_Check: Passed</b>
            &nbsp;&nbsp;<span style="color:#888;font-size:12px;">Articles enforced: {gdpr_articles}</span>
            </div>""",
            unsafe_allow_html=True,
        )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**ORIGINAL** — Biometric Data Exposed ⚠️")
        st.caption("Contains facial geometry · iris patterns · skin-texture fingerprints")
        st.image(original_img, use_container_width=True)
    with col2:
        st.markdown("**GAN SYNTHESIZED** — Biometric Identity Destroyed ✅")
        st.caption("DCGAN generator · PatchGAN discriminator · FGSM adversarial · CycleGAN Instance Norm · Gaussian blur")
        st.image(synth_img, use_container_width=True)

    buf = io.BytesIO()
    synth_img.save(buf, format="PNG")
    synth_bytes = buf.getvalue()

    # ── Compute per-channel metrics for insights pages ────────────────────────
    orig_arr = np.array(original_img.convert("RGB"), dtype=np.float32)
    synth_arr = np.array(synth_img.resize(original_img.size, Image.LANCZOS).convert("RGB"), dtype=np.float32)
    channel_stats = []
    for i, ch_name in enumerate(["R", "G", "B"]):
        oc = orig_arr[:, :, i].flatten()
        sc = synth_arr[:, :, i].flatten()
        ks_stat, ks_p = ks_2samp(oc[::max(1, len(oc)//2000)], sc[::max(1, len(sc)//2000)])
        step = max(1, len(oc) // 600)
        channel_stats.append({
            "channel": ch_name,
            "orig_mean": float(oc.mean()), "orig_std": float(oc.std()),
            "synth_mean": float(sc.mean()), "synth_std": float(sc.std()),
            "ks_statistic": float(ks_stat), "p_value": float(ks_p),
            "passed": ks_stat > 0.15,
            "orig_sample": oc[::step].tolist(), "synth_sample": sc[::step].tolist(),
        })
    mse = float(np.mean((orig_arr - synth_arr) ** 2))
    psnr = float(10 * np.log10((255 ** 2) / (mse + 1e-8)))
    pixel_destruction = float(np.mean(np.abs(orig_arr - synth_arr)) / 255 * 100)

    st.session_state["image_result"] = {
        "filename": uploaded.name,
        "width": original_img.width, "height": original_img.height,
        "channel_stats": channel_stats,
        "mse": mse, "psnr_proxy": psnr,
        "pixel_destruction_pct": pixel_destruction,
        "gdpr_status": gdpr_status, "gdpr_articles": gdpr_articles,
        "gan_stages": [
            ("1", "GDPR Metadata Strip", "EXIF / GPS / ICC / XMP purge — Art. 5(1)(c)", "#f59e0b"),
            ("2", "Instance Normalisation", "CycleGAN-style per-channel normalisation — Ulyanov 2016", "#a855f7"),
            ("3", "DCGAN Generator", "SVD latent encoder + Z-noise + AdaIN + residual decode — Radford 2015 / Karras 2019", "#22c55e"),
            ("4", "PatchGAN Discriminator", "Per-patch realness scoring via contrast + edge energy — Isola pix2pix 2017", "#06b6d4"),
            ("5", "FGSM Adversarial", "Discriminator-guided pixel perturbation — Goodfellow 2014", "#ef4444"),
            ("6", "Gaussian Blur σ≥15", "Facial geometry / landmark destruction", "#3b82f6"),
            ("7", "Colour-Jitter", "Skin-texture & iris-pattern fingerprint removal", "#8b5cf6"),
        ],
        "original_bytes": image_bytes, "synthesized_bytes": synth_bytes,
    }

    st.download_button(
        "Download Synthesized Image (GDPR Clean)",
        data=synth_bytes,
        file_name=f"gdpr_synthesized_{uploaded.name.rsplit('.', 1)[0]}.png",
        mime="image/png",
        use_container_width=True,
    )
    st.info(
        "**GAN Pipeline (GDPR Art. 9):** EXIF/GPS/ICC purged → "
        "**CycleGAN Instance Norm** (Ulyanov 2016) → "
        "**DCGAN Generator** via SVD latent space + Z-noise + AdaIN style injection (Radford 2015 / Karras 2019) → "
        "**PatchGAN Discriminator** per-patch realness scoring (Isola pix2pix 2017) → "
        "**FGSM Adversarial Perturbation** discriminator-guided pixel attack (Goodfellow 2014) → "
        "Gaussian blur σ≥15 destroys facial geometry → Colour-jitter removes skin-texture fingerprints. "
        "Original biometric footprint irreversibly destroyed. "
        "→ **Navigate to Gen AI Insights, Fidelity Report, Comparison Charts in the sidebar for full analytics.**"
    )
    st.stop()

# ── AUDIO ─────────────────────────────────────────────────────────────────────
elif ext in ("wav", "mp3"):
    audio_bytes = uploaded.read()
    st.success(f"Loaded **{uploaded.name}** — {len(audio_bytes) / 1024:.1f} KB audio file")

    st.markdown("**Original Audio** — Voice-print biometric data present ⚠️")
    st.audio(audio_bytes, format=f"audio/{ext}")

    with st.spinner("Running GDPR voice-print destruction pipeline..."):
        try:
            resp = requests.post(
                f"{API_URL}/api/process/audio",
                files={"file": (uploaded.name, audio_bytes, f"audio/{ext}")},
                timeout=60,
            )
            resp.raise_for_status()
            synthesized_wav = resp.content
            gdpr_status = resp.headers.get("X-GDPR-Compliance-Check", "Unknown")
            gdpr_articles = resp.headers.get("X-GDPR-Articles", "")
            sample_rate = resp.headers.get("X-Sample-Rate", "")
        except Exception as e:
            st.error(f"Backend error: {e}")
            st.stop()

    if gdpr_status == "Passed":
        st.markdown(
            f"""<div style="border:1px solid #22c55e;border-radius:6px;padding:8px 14px;
            background:#22c55e15;display:inline-block;margin:8px 0;">
            ✅ <b style="color:#22c55e;">GDPR_Compliance_Check: Passed</b>
            &nbsp;&nbsp;<span style="color:#888;font-size:12px;">Articles enforced: {gdpr_articles}</span>
            {f'&nbsp;·&nbsp;<span style="color:#888;font-size:12px;">Sample rate: {sample_rate} Hz</span>' if sample_rate else ''}
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("**GAN Synthesized Audio** — Voice-print irreversibly destroyed ✅")
    st.caption("WaveGAN generator · StarGAN-VC formant warp · MelGAN 3-scale discriminator · FGSM adversarial · F0 pitch-shift")
    st.audio(synthesized_wav, format="audio/wav")

    # ── Compute audio metrics for insights pages ──────────────────────────────
    try:
        import soundfile as _sf
    except Exception:
        _sf = None
    try:
        orig_wav, sr_load = _sf.read(io.BytesIO(audio_bytes))
        synth_wav, _ = _sf.read(io.BytesIO(synthesized_wav))
        if orig_wav.ndim > 1: orig_wav = orig_wav[:, 0]
        if synth_wav.ndim > 1: synth_wav = synth_wav[:, 0]
        min_len = min(len(orig_wav), len(synth_wav))
        ow = orig_wav[:min_len].astype(np.float32)
        sw = synth_wav[:min_len].astype(np.float32)
        orig_rms = float(np.sqrt(np.mean(ow ** 2)) + 1e-9)
        synth_rms = float(np.sqrt(np.mean(sw ** 2)) + 1e-9)
        orig_zcr = float(np.mean(np.abs(np.diff(np.sign(ow)))) / 2)
        synth_zcr = float(np.mean(np.abs(np.diff(np.sign(sw)))) / 2)
        nf = min(2048, min_len)
        of = np.abs(np.fft.rfft(ow[:nf]))
        sf2 = np.abs(np.fft.rfft(sw[:nf]))
        freqs = np.fft.rfftfreq(nf, 1.0 / (int(sample_rate) if sample_rate else 22050)).tolist()
        eps = 1e-10
        on = of / (of.sum() + eps)
        sn = sf2 / (sf2.sum() + eps)
        spec_div = float(np.sum(on * np.log(on / (sn + eps) + eps)))
        vp_score = float(min(100, spec_div * 12 + abs(orig_rms - synth_rms) / orig_rms * 35))
        fstep = max(1, len(freqs) // 500)
        wstep = max(1, min_len // 1000)
        audio_metrics_ok = True
    except Exception:
        audio_metrics_ok = False
        orig_rms = synth_rms = orig_zcr = synth_zcr = spec_div = vp_score = 0.0

    if audio_metrics_ok:
        st.session_state["audio_result"] = {
            "filename": uploaded.name,
            "sample_rate": int(sample_rate) if sample_rate else 22050,
            "duration_s": float(min_len / (int(sample_rate) if sample_rate else 22050)),
            "orig_rms": orig_rms, "synth_rms": synth_rms,
            "orig_zcr": orig_zcr, "synth_zcr": synth_zcr,
            "spectral_divergence": spec_div,
            "voice_print_destruction_pct": vp_score,
            "orig_waveform": ow[::wstep].tolist(),
            "synth_waveform": sw[::wstep].tolist(),
            "orig_spectrum": of[::fstep].tolist(),
            "synth_spectrum": sf2[::fstep].tolist(),
            "freqs": freqs[::fstep],
            "gdpr_status": gdpr_status, "gdpr_articles": gdpr_articles,
            "gan_stages": [
                ("1", "GDPR Metadata Strip", "ID3 / Vorbis / RIFF INFO purge — Art. 5(1)(c)", "#f59e0b"),
                ("2", "WaveGAN Generator", "STFT-DCT latent encoder — Donahue et al. 2018", "#22c55e"),
                ("3", "StarGAN-VC Voice Conversion", "Formant-bin warping F1–F4 — Kameoka et al. 2018", "#06b6d4"),
                ("4", "Z-Noise Injection", "z ~ N(0,σ) in frequency-domain latent space", "#a855f7"),
                ("5", "tanh Activation", "GAN generator output layer — prevents amplitude explosion", "#8b5cf6"),
                ("6", "IDCT + OLA Decoder", "Overlap-add reconstruction with Hann window", "#3b82f6"),
                ("7", "MelGAN Discriminator", "3-scale temporal discriminator — Kumar et al. 2019", "#ef4444"),
                ("8", "FGSM Adversarial", "Discriminator-weighted noise injection", "#f97316"),
                ("9", "F0 Pitch Shift ±18%", "Fundamental frequency destruction", "#ec4899"),
                ("10", "Band-Limited Noise", "Residual speaker characteristic masking", "#14b8a6"),
            ],
            "original_bytes": audio_bytes, "synthesized_bytes": synthesized_wav,
        }

    st.download_button(
        "Download Synthesized Audio (GDPR Clean WAV)",
        data=synthesized_wav,
        file_name=f"gdpr_synthesized_{uploaded.name.rsplit('.', 1)[0]}.wav",
        mime="audio/wav",
        use_container_width=True,
    )
    st.info(
        "**GAN Pipeline (GDPR Art. 9):** ID3/Vorbis/RIFF metadata purged → "
        "**WaveGAN Generator** STFT-DCT latent encoder + StarGAN-VC formant-bin warping + Z-noise injection + tanh activation + IDCT decoder (Donahue 2018 / Kameoka 2018) → "
        "**MelGAN Multi-Scale Discriminator** evaluates waveform at 3 temporal scales (Kumar 2019) → "
        "**FGSM Adversarial** discriminator-weighted noise pushes segments into synthetic distribution → "
        "F0 pitch-shift ±18% destroys fundamental frequency fingerprint → "
        "Band-limited noise masks residual speaker characteristics. "
        "Speaker-recognition systems cannot match output to original voice print. "
        "→ **Navigate to Gen AI Insights, Fidelity Report, Comparison Charts in the sidebar for full analytics.**"
    )
    st.stop()

# ── CSV Results Guard ──────────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.stop()

result = st.session_state["result"]
orig_df = st.session_state.get("orig_df", pd.DataFrame())

# ── Results Summary ───────────────────────────────────────────────────────────
st.divider()
st.subheader("Results")

r1, r2, r3, r4 = st.columns(4)
r1.metric("Re-ID Risk Score", f"{result['risk_score']:.1f}%",
          delta=f"{result['risk_score']-50:.1f}% vs threshold", delta_color="inverse")
r2.metric("PII Fields Detected", len(result["ner_detections"]))
r3.metric("ε Budget Used", f"{result['epsilon_budget']['spent']:.4f} / {result['epsilon_budget']['total']}")
r4.metric("GAN Epochs", result["gan_epochs"])

# ── Three-way data table ───────────────────────────────────────────────────────
tab_orig, tab_anon, tab_syn = st.tabs(["Original Data", "Anonymized (K-Anon + DP)", "Synthetic (CTGAN)"])
with tab_orig:
    st.dataframe(pd.DataFrame(result["original_preview"]), use_container_width=True, hide_index=True)
with tab_anon:
    st.dataframe(pd.DataFrame(result["anonymized_preview"]), use_container_width=True, hide_index=True)
with tab_syn:
    st.dataframe(pd.DataFrame(result["synthetic_preview"]), use_container_width=True, hide_index=True)

# ── Downloads ─────────────────────────────────────────────────────────────────
st.subheader("Downloads")
d1, d2, d3 = st.columns(3)
with d1:
    st.download_button("Download Anonymized CSV", data=result["anonymized_csv"],
                       file_name=f"anonymized_{uploaded.name}", mime="text/csv", use_container_width=True)
with d2:
    st.download_button("Download Synthetic CSV (CTGAN)", data=result["synthetic_csv"],
                       file_name=f"synthetic_{uploaded.name}", mime="text/csv", use_container_width=True)
with d3:
    summary = f"""ANONYMITY ENGINE — SESSION REPORT
==============================
File: {uploaded.name}
Epsilon: {epsilon}
K-Anonymity: {k_value}
Risk Score: {result['risk_score']:.1f}%
PII Detected: {len(result['ner_detections'])} fields
GAN Epochs: {result['gan_epochs']}
Fidelity Verdict: {result['fidelity_report']['verdict']}
Overall Fidelity: {result['fidelity_report']['overall_fidelity']:.1f}%
Utility Score: {result['fidelity_report']['utility_score']:.1f}%
Privacy Score: {result['fidelity_report']['privacy_score']:.1f}%
"""
    st.download_button("Download Session Report", data=summary,
                       file_name="session_report.txt", mime="text/plain", use_container_width=True)

