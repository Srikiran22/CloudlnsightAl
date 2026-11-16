@echo off
setlocal
cd /d "%~dp0"

echo ========================================================
echo Starting TabuLens Streamlit Application...
echo ========================================================

:: Check for virtual environment in venv or .venv
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    echo [WARNING] No virtual environment found in .\venv or .\.venv.
    echo Running with system Python...
)

:: Run streamlit app
python -m streamlit run app.py

pause
