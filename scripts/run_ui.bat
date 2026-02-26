@echo off
:: BayanSynthTTS — Launch Gradio UI
:: Opens the inference web interface at http://localhost:7865

setlocal
set "BAYAN_DIR=%~dp0.."
set "REPO_ROOT=%~dp0..\.."
set "VENV_PY=%REPO_ROOT%\.venv\Scripts\python.exe"
set "APP_PY=%BAYAN_DIR%\bayansynthtts\app.py"

if not exist "%VENV_PY%" (
    echo [run_ui] ERROR: venv not found at:
    echo           %VENV_PY%
    echo         Create it from the repo root:
    echo           python -m venv .venv
    echo           .venv\Scripts\pip install -r BayanSynthTTS\requirements.txt
    exit /b 1
)

if not exist "%APP_PY%" (
    echo [run_ui] ERROR: app.py not found at %APP_PY%
    exit /b 1
)

cd /d "%REPO_ROOT%"
echo Starting BayanSynthTTS UI at http://localhost:7865 ...
"%VENV_PY%" "%APP_PY%" --port 7865
