@echo off
:: BayanSynthTTS — Quick Inference via CLI
::
:: Usage:
::   infer.bat "مَرْحَباً بِكُمْ"
::   infer.bat "مَرْحَباً" --output hello.wav --voice voices\my_voice.wav
::   infer.bat --help

setlocal
set "BAYAN_DIR=%~dp0.."
set "REPO_ROOT=%~dp0..\.."
set "VENV_PY=%REPO_ROOT%\.venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
    echo [infer] ERROR: venv not found at:
    echo           %VENV_PY%
    echo         Create it from the repo root:
    echo           python -m venv .venv
    echo           .venv\Scripts\pip install -r BayanSynthTTS\requirements.txt
    exit /b 1
)

cd /d "%REPO_ROOT%"
"%VENV_PY%" -m bayansynthtts.inference %*
