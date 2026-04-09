# Anonymity Engine

A full-stack Python application for automated data anonymization and synthetic data generation, implementing a 5-Agent Gen AI pipeline.

## Architecture

- **Frontend**: Streamlit multipage app on port 5000 (0.0.0.0)
- **Backend**: FastAPI REST API on port 8000 (localhost)
- **Database**: PostgreSQL (Replit built-in, via `DATABASE_URL`)

## Project Layout

```
app.py                  # Streamlit Command Center (main entry point)
start.sh                # Startup script (launches backend + frontend)
privacy_engine.py       # Core library: DP, NER, CTGAN, KS-test logic
backend/
  main.py               # FastAPI routes and API logic
  database.py           # SQLAlchemy engine and session setup
  models.py             # ORM models (anonymization_sessions table)
pages/
  1_Gen_AI_Insights.py  # NER detections and Epsilon budget tracker
  2_Fidelity_Report.py  # KS tests and Judge verdict reports
  3_Comparison_Charts.py# Distribution comparison visualizations
  4_Secure_Vault.py     # Historical session management
.streamlit/config.toml  # Streamlit server config (port 5000, all hosts)
requirements.txt        # Python dependencies
```

## 5-Agent Pipeline

1. **The Sentry** — Zero-Shot NER: Detects PII via regex and semantic header analysis
2. **The Auditor** — Risk Modeling: Analyzes re-identification risks and quasi-identifiers
3. **The Ghost** — Differential Privacy: Applies Laplace/Exponential mechanisms
4. **The Alchemist** — CTGAN/Gaussian Copula: Generates synthetic data
5. **The Judge** — Fidelity Report: Validates quality using KS tests

## Running

```bash
bash start.sh
```

This starts:
- FastAPI backend: `uvicorn backend.main:app --host localhost --port 8000`
- Streamlit frontend: `streamlit run app.py` on port 5000

## Environment Variables

- `DATABASE_URL` — PostgreSQL connection string (auto-set by Replit)
- `BACKEND_URL` — FastAPI backend URL (defaults to `http://localhost:8000`)

## Database

Table: `anonymization_sessions`
- Tracks all processing sessions (CSV, image, audio)
- Stores privacy budget, k-anonymity settings, risk scores
