@echo off
:: BayanSynthTTS — Model Setup
:: Downloads CosyVoice3 base model and LoRA checkpoints.

setlocal
set "BAYAN_DIR=%~dp0.."
set "SETUP_PY=%BAYAN_DIR%\scripts\setup_models.py"

:: Look for venv inside BayanSynthTTS/ first (standalone), then parent dir (dev)
if exist "%BAYAN_DIR%\.venv\Scripts\python.exe" (
    set "VENV_PY=%BAYAN_DIR%\.venv\Scripts\python.exe"
) else if exist "%BAYAN_DIR%\..\..venv\Scripts\python.exe" (
    set "VENV_PY=%BAYAN_DIR%\..\..venv\Scripts\python.exe"
) else if exist "%BAYAN_DIR%\..\.venv\Scripts\python.exe" (
    set "VENV_PY=%BAYAN_DIR%\..\.venv\Scripts\python.exe"
) else (
    echo [setup] ERROR: No virtual environment found.
    echo         Create one inside BayanSynthTTS/:
    echo           cd BayanSynthTTS
    echo           python -m venv .venv
    echo           .venv\Scripts\pip install -r requirements.txt
    echo           .venv\Scripts\pip install -e .
    exit /b 1
)

if not exist "%SETUP_PY%" (
    echo [setup] ERROR: setup_models.py not found at %SETUP_PY%
    exit /b 1
)

cd /d "%BAYAN_DIR%"
"%VENV_PY%" "%SETUP_PY%" %*
