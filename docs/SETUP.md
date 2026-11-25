# TabuLens — Setup & Local Execution Guide

This guide covers setting up, running, and troubleshooting TabuLens locally or inside Docker.

---

## 1. Prerequisites

- **OS:** Windows 10/11, macOS 13+, or modern Linux
- **Python:** Python **3.11** or newer (`3.11` and `3.12` are tested in CI; `pyproject.toml` declares `>=3.11`)
- **Git:** Optional for cloning
- **Docker (Optional):** Docker Engine and Docker Compose v2

---

## 2. Local Installation

### 2.1 Create and Activate a Virtual Environment

```bash
git clone https://github.com/Srikiran22/CloudlnsightAl.git
cd CloudlnsightAl

python -m venv .venv
```

Activate it:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Windows Command Prompt
.\.venv\Scripts\activate.bat

# macOS / Linux
source .venv/bin/activate
```

*(On Windows PowerShell, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` if script activation is restricted.)*

### 2.2 Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

To install exact locked transitive versions instead:

```bash
pip install -r requirements-lock.txt
```

---

## 3. Running the Application

### Standard Launch

```bash
streamlit run app.py
```

On Windows, `start.bat` activates `.venv` (or `venv`) if present and runs `python -m streamlit run app.py`.

Open `http://127.0.0.1:8501` in your browser. Click **Load sample datasets (SaaS Churn, Drift & Housing)** on the home view to populate synthetic datasets into `Datasets/` without external downloads.

### Docker Compose Launch

```bash
docker compose up --build
```

Docker Compose builds the two-stage `Dockerfile` (`python:3.11-slim`, non-root `appuser`), binds port `127.0.0.1:8501:8501`, mounts `./Datasets`, `./Models`, and `./Reports`, and monitors `/_stcore/health` with `scripts/healthcheck.py`.

### Diagnostic Logging

Set `TABULENS_LOG_LEVEL` (`DEBUG`, `INFO`, `WARNING`, `ERROR`; default `WARNING`) to control stderr logs emitted by `utils/logsys.py`:

```bash
TABULENS_LOG_LEVEL=INFO streamlit run app.py
```

---

## 4. Verification

```bash
# Static layout, dependency, and HMAC verification checks
python scripts/verify_pipeline.py --check-only

# Full static checks + unittest suite
python scripts/verify_pipeline.py

# Standard unittest discovery
python -m unittest discover -s tests -v

# Linter
ruff check .
```

---

## 5. Troubleshooting

| Symptom | Cause | Solution |
|---|---|---|
| `'python' is not recognized` | Python missing from `PATH` | Install Python 3.11+ and add it to `PATH`, or invoke the `.venv` interpreter directly. |
| `ValueError` on model load (`Untrusted model file` / `Cryptographic signature verification failed`) | `.joblib` file lacks `.joblib.sig`, was modified on disk, or was signed with a different `Models/.model_signing_key` | Retrain and save the model in **Model Studio & Training** so a valid `.joblib` + `.joblib.sig` pair is created for the current signing key. |
| Standalone script exits with signature error | Key not found next to the model or in `Models/` | Pass `--key-path /path/to/.model_signing_key` or set `TABULENS_SIGNING_KEY_PATH`. |
| Port `8501` already in use | Another Streamlit instance is running | Run `streamlit run app.py --server.port 8502`. |
