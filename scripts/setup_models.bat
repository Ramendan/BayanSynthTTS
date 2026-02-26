@echo off
:: BayanSynthTTS — Model Setup
:: Checks base model, LoRA checkpoints, and default voice.

setlocal
set "REPO_ROOT=%~dp0..\.."
set "VENV_PY=%REPO_ROOT%\.venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
    echo [setup] venv not found at %VENV_PY%
    echo         Run this from the CosyVoice repo root after creating the venv.
    exit /b 1
)

cd /d "%REPO_ROOT%"
"%VENV_PY%" BayanSynthTTS\scripts\setup_models.py %*
