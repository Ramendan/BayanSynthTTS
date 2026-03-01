@echo off
:: BayanSynthTTS — Launch Gradio UI
:: Opens the inference web interface at http://localhost:7865

setlocal
set "BAYAN_DIR=%~dp0.."
set "APP_PY=%BAYAN_DIR%\bayansynthtts\app.py"

:: Look for venv inside BayanSynthTTS/ first (standalone), then parent dir (dev)
if exist "%BAYAN_DIR%\.venv\Scripts\python.exe" (
    set "VENV_PY=%BAYAN_DIR%\.venv\Scripts\python.exe"
) else if exist "%BAYAN_DIR%\..\..venv\Scripts\python.exe" (
    set "VENV_PY=%BAYAN_DIR%\..\..venv\Scripts\python.exe"
) else if exist "%BAYAN_DIR%\..\.venv\Scripts\python.exe" (
    set "VENV_PY=%BAYAN_DIR%\..\.venv\Scripts\python.exe"
) else (
    echo [run_ui] ERROR: No virtual environment found.
    echo         Create one inside BayanSynthTTS/:
    echo           cd BayanSynthTTS
    echo           python -m venv .venv
    echo           .venv\Scripts\pip install -r requirements.txt
    echo           .venv\Scripts\pip install -e .
    exit /b 1
)

if not exist "%APP_PY%" (
    echo [run_ui] ERROR: app.py not found at %APP_PY%
    exit /b 1
)

cd /d "%BAYAN_DIR%"
echo Starting BayanSynthTTS UI at http://localhost:7865 ...
"%VENV_PY%" "%APP_PY%" --port 7865
