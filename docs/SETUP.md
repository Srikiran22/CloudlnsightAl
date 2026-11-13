# CloudInsight AI — Setup & Local Execution Guide

This document provides step-by-step instructions to set up, configure, run, and troubleshoot CloudInsight AI on a local workstation.

---

## 1. Prerequisites

- **Operating System:** Windows 10/11 (primary verified platform), macOS 13+, or modern Linux (Ubuntu 22.04+).
- **Python Version:** Python **3.12** or **3.13** (64-bit).
  - *Verify installation:*
    ```bash
    python --version
    ```
  - *Windows Installation via winget:*
    ```powershell
    winget install Python.Python.3.12
    ```
- **Git:** Git 2.40+ (local execution only; no remote access required).

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

For an exact, reproducible installation identical to our verified test baseline, install using the pinned lockfile:

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements-lock.txt
```

Alternatively, to install using flexible version ranges:

```bash
pip install -r requirements.txt
```

---

## 3. Running the Application

### 3.1 Quick Start (Windows)
Double-click `start.bat` or run:
```bat
start.bat
```

### 3.2 Standard CLI Launch
With your virtual environment activated:
```bash
python -m streamlit run App.py
```

Streamlit will print the local server URL:
```
  Local URL: http://localhost:8501
  Network URL: http://127.0.0.1:8501
```
Open your browser to `http://localhost:8501`.

---

## 4. Running Verification Checks

### 4.1 Pytest Suite Offline Execution (Recommended Primary Runner)
Run the full 8-module test harness locally without internet access or live API keys:
```bash
pytest tests/ -v
```

Run with test coverage breakdown:
```bash
pytest tests/ --cov=Utils --cov=Pages --cov=App --cov-report=term-missing
```

### 4.2 Headless Page Smoke Tests
Verify that all 13 application pages boot cleanly without unhandled exceptions:
```bash
pytest tests/test_page_smoke.py -v
```

### 4.3 Standard Library Unittest Fallback (Zero Dependency)
```bash
python -m unittest discover -s tests -v
python -m unittest tests/test_page_smoke.py -v
```

### 4.3 Code Quality & Linting
Run Ruff static analysis:
```bash
pip install ruff==0.16.4
ruff check .
```

---

## 5. Configuration & Credentials

### 5.1 Streamlit Native Theming (`.streamlit/config.toml`)
The application includes a mirrored design system configured in `.streamlit/config.toml`:
- Follows OS/browser preference (light or dark) by default.
- Users can manually switch modes via the top-right **⋮ (menu) → Settings → Theme**.

### 5.2 Google Gemini Integration (Optional)
Gemini is completely optional. Core ingestion, profiling, cleaning, EDA, visualization, ML training, cross-validation, and PDF export function 100% locally without external APIs.
- When using AI File Converter or AI Summary:
  1. Obtain a key from [Google AI Studio](https://aistudio.google.com/).
  2. Input the key into the **Google Gemini API key** field in the sidebar.
  3. Keys remain in ephemeral session memory (`st.session_state`) only.

---

## 6. Troubleshooting Common Issues

| Symptom | Cause | Solution |
|---|---|---|
| `'python' is not recognized` / Windows Store prompt | Python is not installed or not in PATH | Install Python 3.12+ via winget (`winget install Python.Python.3.12`) and ensure "Add Python to PATH" is checked. |
| `PermissionError` on model load | Model file was modified or unsigned | CloudInsight verifies HMAC-SHA256 signatures before loading. Retrain the model in **Model Studio** to produce a freshly signed `.joblib` and `.sig` pair. |
| `xlrd` missing when loading `.xls` | Legacy Excel file format (.xls) uploaded | Install `xlrd`: `pip install xlrd>=2.0.1` or save the spreadsheet as modern `.xlsx` or `.csv`. |
| Port 8501 already in use | Another Streamlit process is running | Run on a different port: `streamlit run App.py --server.port 8502`. |
