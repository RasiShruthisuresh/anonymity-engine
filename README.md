# ANONYMITY ENGINE

Full-stack Python data anonymisation platform with a **5-agent Gen AI pipeline** for CSV data and GAN-powered multimodal pipelines for **images** and **audio** — all under GDPR Art. 5/9/17 compliance.

---

## Architecture

| Layer | Technology | Port |
|-------|-----------|------|
| Frontend | Streamlit multipage app | 5000 |
| Backend | FastAPI REST API | 8000 |
| Database | PostgreSQL via SQLAlchemy ORM | 5432 |

---

## Project Structure

```
anonymity-engine/
├── app.py                        # Streamlit Command Center (main page)
├── start.sh                      # Launch script (uvicorn + streamlit)
├── requirements.txt              # All Python dependencies
├── privacy_engine.py             # Core privacy library
│   ├── CSV pipeline              # NER · DP · K-Anon · CTGAN · KS-test
│   ├── Image GAN pipeline        # DCGAN · CycleGAN · PatchGAN · FGSM
│   └── Audio GAN pipeline        # WaveGAN · MelGAN · StarGAN-VC · FGSM
├── backend/
│   ├── main.py                   # FastAPI routes
│   ├── models.py                 # SQLAlchemy ORM models
│   └── database.py               # DB connection
└── pages/
    ├── 1_Gen_AI_Insights.py      # GAN architecture details, NER, DP tracker
    ├── 2_Fidelity_Report.py      # KS-test, radar chart, verdict
    ├── 3_Comparison_Charts.py    # Distribution / waveform / spectrum charts
    └── 4_Secure_Vault.py         # PostgreSQL session history
```

---

## Gen AI Features

### CSV — 5-Agent Pipeline

| Agent | Role | Technology |
|-------|------|-----------|
| **Sentry** | Zero-Shot Transformer NER | Semantic header analysis + regex PII detection; simulates BERT/RoBERTa confidence scores |
| **Ghost** | Differential Privacy | Laplace mechanism (continuous columns) + Exponential mechanism (categorical); epsilon budget tracker |
| **Phantom** | K-Anonymity | Quasi-identifier suppression and generalisation |
| **Alchemist** | CTGAN Gaussian Copula | Pearson correlation matrix → Cholesky decomposition → marginal inverse-CDF transform → DP noise injection |
| **Judge** | Fidelity Evaluation | scipy `ks_2samp` KS test; utility vs privacy radar; PASS / WARN / FAIL verdict |

### Image — GAN Biometric Destruction Pipeline (GDPR Art. 9)

| Stage | Algorithm | Reference |
|-------|-----------|-----------|
| Metadata Strip | EXIF / GPS / ICC / XMP purge | GDPR Art. 5(1)(c) |
| Instance Normalisation | CycleGAN-style per-channel norm | Ulyanov et al. 2016 |
| DCGAN Generator | SVD latent encoder + Z-noise + AdaIN style injection | Radford 2015, Karras 2019 |
| PatchGAN Discriminator | Per-patch realness scoring via contrast + edge energy | Isola pix2pix 2017 |
| FGSM Adversarial | Discriminator-guided pixel perturbation | Goodfellow et al. 2014 |
| Gaussian Blur σ≥15 | Facial geometry / landmark destruction | — |
| Colour-Jitter | Skin-texture and iris-pattern fingerprint removal | — |

**Metrics computed:** per-channel KS distance (R/G/B), MSE, PSNR proxy, pixel destruction %.

### Audio — GAN Voice-Print Destruction Pipeline (GDPR Art. 9)

| Stage | Algorithm | Reference |
|-------|-----------|-----------|
| Metadata Strip | ID3 / Vorbis / RIFF INFO purge | GDPR Art. 5(1)(c) |
| WaveGAN Generator | STFT-DCT latent encoder | Donahue et al. 2018 |
| StarGAN-VC | Formant-bin warping F1–F4 | Kameoka et al. 2018 |
| Z-Noise Injection | z ~ N(0,σ) in frequency-domain latent space | — |
| tanh Activation | GAN generator output layer | — |
| IDCT + OLA Decoder | Overlap-add reconstruction with Hann window | — |
| MelGAN Discriminator | 3-scale temporal discriminator | Kumar et al. 2019 |
| FGSM Adversarial | Discriminator-weighted noise injection | Goodfellow et al. 2014 |
| F0 Pitch Shift ±18% | Fundamental frequency destruction | — |
| Band-Limited Noise | Residual speaker characteristic masking | — |

**Metrics computed:** RMS energy, zero-crossing rate, FFT spectrum, spectral divergence (KL-like), voice-print destruction score.

---

## Analytics Pages (Sidebar)

All three analytics pages support **all three media types** (CSV / Image / Audio) via tabbed interface.

| Page | CSV | Image | Audio |
|------|-----|-------|-------|
| **Gen AI Insights** | NER detections, ε budget, CTGAN stats | GAN stage cards, per-channel KS, GDPR record | WaveGAN stage cards, RMS/ZCR metrics, GDPR record |
| **Fidelity Report** | KS-test results, radar chart, Judge verdict | Pixel destruction metrics, biometric radar, score breakdown | Spectral divergence, voice-print radar, Judge verdict |
| **Comparison Charts** | Distribution histograms, mean comparison, KS heatmap | RGB channel histograms (orig vs synth), mean bar chart | Waveform comparison, FFT spectrum, MelGAN 3-scale discriminator |

---

## API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/health` | Health check |
| GET | `/api/sessions` | List all processing sessions |
| POST | `/api/sessions` | Create a session record |
| GET | `/api/sessions/stats` | Aggregate stats across sessions |
| DELETE | `/api/sessions/{id}` | Delete a session |
| POST | `/api/process/csv` | Run full 5-agent pipeline on uploaded CSV |
| POST | `/api/process/image` | Run GAN biometric destruction on image |
| POST | `/api/process/audio` | Run WaveGAN voice-print destruction on audio |
| GET | `/api/gdpr/system-prompt` | Retrieve the GDPR system prompt used by the pipeline |

### Response Headers (image and audio endpoints)

```
X-GDPR-Compliance-Check: Passed
X-GDPR-Articles: Art5,Art9,Art17
X-Sample-Rate: 22050          # audio only
```

---

## GDPR Compliance

| Article | Enforcement |
|---------|-------------|
| **Art. 5(1)(c) — Data Minimisation** | All file metadata (EXIF, ID3, GPS) stripped before processing |
| **Art. 9 — Special Category Data** | Facial geometry, iris patterns, voice prints irreversibly destroyed |
| **Art. 17 — Right to Erasure** | Originals processed in-memory only; nothing persisted to disk |

---

## Local Setup

```bash
# 1. Clone / unzip the project
cd anonymity-engine

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set environment variables
export DATABASE_URL="postgresql://postgres:yourpassword@localhost:5432/anonymity_engine"
export BACKEND_URL="http://localhost:8000"

# 4. Terminal 1 — start the backend (tables auto-created)
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# 5. Terminal 2 — start the frontend
streamlit run app.py --server.port 5000

# Or simply:
bash start.sh
```

Open: http://localhost:5000  
API docs: http://localhost:8000/docs

---

## Supported File Formats

| Type | Extensions |
|------|-----------|
| Tabular data | `.csv` |
| Images | `.jpg`, `.jpeg`, `.png`, `.webp` |
| Audio | `.wav`, `.mp3` |
