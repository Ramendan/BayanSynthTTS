@echo off
:: BayanSynthTTS — Launch Gradio UI
:: Opens the inference web interface at http://localhost:7865

setlocal
set "REPO_ROOT=%~dp0..\.."
set "VENV_PY=%REPO_ROOT%\.venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
    echo [run_ui] venv not found at %VENV_PY%
    exit /b 1
)

cd /d "%REPO_ROOT%"
echo Starting BayanSynthTTS UI at http://localhost:7865
"%VENV_PY%" BayanSynthTTS\bayansynthtts\app.py --port 7865
