@echo off
:: BayanSynthTTS — Quick Inference via CLI
::
:: Usage:
::   infer.bat "مَرْحَباً بِكُمْ"
::   infer.bat "مَرْحَباً" --output hello.wav --voice voices\my_voice.wav
::   infer.bat --help

setlocal
set "REPO_ROOT=%~dp0..\.."
set "VENV_PY=%REPO_ROOT%\.venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
    echo [infer] venv not found at %VENV_PY%
    exit /b 1
)

cd /d "%REPO_ROOT%"
"%VENV_PY%" -m bayansynthtts.inference %*
