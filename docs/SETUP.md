# TabuLens — Setup & Local Execution Guide

This document provides step-by-step instructions to set up, configure, run, and troubleshoot TabuLens on a local workstation or container environment.

---

## 1. Prerequisites

- **Operating System:** Windows 10/11, macOS 13+, or modern Linux (Ubuntu 22.04+).
- **Python Version:** Python **3.12** or newer (64-bit).
  - *Verify installation:*
    ```bash
    python --version
    ```
- **Git:** Git 2.40+ (local execution only; no remote access required).
- **Docker (Optional):** Docker Engine 24+ and Docker Compose v2 for containerized execution.

---

## 2. Environment Installation

### 2.1 Create a Virtual Environment

Open a terminal in the project directory:

```bash
# Windows (PowerShell or Command Prompt)
python -m venv venv

# macOS / Linux
python3 -m venv venv
```

### 2.2 Activate the Environment

```powershell
# Windows PowerShell
.\venv\Scripts\Activate.ps1

# Windows Command Prompt
.\venv\Scripts\activate.bat

# macOS / Linux
source venv/bin/activate
```

*(Note for Windows PowerShell: If script execution is restricted, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` before activating).*

### 2.3 Install Dependencies

Install using the project dependencies:

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

## 3. Running the Application

### 3.1 Standard Streamlit Launch

With your virtual environment activated:

```bash
streamlit run app.py
```

Streamlit will print the local server URL:
```
  Local URL: http://localhost:8501
  Network URL: http://127.0.0.1:8501
```
Open your browser to `http://localhost:8501`.

### 3.2 Docker Compose Execution

To run TabuLens in an isolated container without local Python dependencies:

```bash
docker compose up --build
```

Access the application in your browser at `http://localhost:8501`.

---

## 4. Running Verification Checks

### 4.1 Verification Runner

Run the end-to-end static environment check and test suite:

```bash
# Full verification (environment, HMAC invariant, test suite)
python scripts/verify_pipeline.py

# Or static configuration check only
python scripts/verify_pipeline.py --check-only
```

### 4.2 Standard Library Unittest Suite

Execute the full discovery test harness:

```bash
python -m unittest discover -s tests -v
```

---

## 5. Troubleshooting Common Issues

| Symptom | Cause | Solution |
|---|---|---|
| `'python' is not recognized` | Python is not installed or not in PATH | Install Python 3.12+ and ensure "Add Python to PATH" is checked during setup. |
| `PermissionError` on model load | Model file was modified or unsigned | TabuLens verifies HMAC-SHA256 signatures before loading. Retrain the model in **Model Studio** to produce a freshly signed `.joblib` and `.sig` pair. |
| `xlrd` missing when loading `.xls` | Legacy Excel file format (.xls) uploaded | Save the spreadsheet as modern `.xlsx` or `.csv`. |
| Port 8501 already in use | Another Streamlit process is running | Run on a different port: `streamlit run app.py --server.port 8502`. |
