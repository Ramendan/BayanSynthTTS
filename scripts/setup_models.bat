@echo off
:: BayanSynthTTS — Model Setup
:: Checks base model, LoRA checkpoints, and default voice.

setlocal
set "BAYAN_DIR=%~dp0.."
set "REPO_ROOT=%~dp0..\.."
set "VENV_PY=%REPO_ROOT%\.venv\Scripts\python.exe"
set "SETUP_PY=%BAYAN_DIR%\scripts\setup_models.py"

if not exist "%VENV_PY%" (
    echo [setup] ERROR: venv not found at:
    echo           %VENV_PY%
    echo         Create it from the repo root:
    echo           python -m venv .venv
    echo           .venv\Scripts\pip install -r BayanSynthTTS\requirements.txt
    exit /b 1
)

if not exist "%SETUP_PY%" (
    echo [setup] ERROR: setup_models.py not found at %SETUP_PY%
    exit /b 1
)

cd /d "%REPO_ROOT%"
"%VENV_PY%" "%SETUP_PY%" %*
