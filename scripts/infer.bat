@echo off
:: BayanSynthTTS — Quick Inference via CLI
::
:: Usage:
::   infer.bat "مَرْحَباً بِكُمْ"
::   infer.bat "مَرْحَباً" --output hello.wav --voice voices\my_voice.wav
::   infer.bat --help

setlocal
set "BAYAN_DIR=%~dp0.."

:: Look for venv inside BayanSynthTTS/ first (standalone), then parent dir (dev)
if exist "%BAYAN_DIR%\.venv\Scripts\python.exe" (
    set "VENV_PY=%BAYAN_DIR%\.venv\Scripts\python.exe"
) else if exist "%BAYAN_DIR%\..\..venv\Scripts\python.exe" (
    set "VENV_PY=%BAYAN_DIR%\..\..venv\Scripts\python.exe"
) else if exist "%BAYAN_DIR%\..\.venv\Scripts\python.exe" (
    set "VENV_PY=%BAYAN_DIR%\..\.venv\Scripts\python.exe"
) else (
    echo [infer] ERROR: No virtual environment found.
    echo         Create one inside BayanSynthTTS/:
    echo           cd BayanSynthTTS
    echo           python -m venv .venv
    echo           .venv\Scripts\pip install -r requirements.txt
    echo           .venv\Scripts\pip install -e .
    exit /b 1
)

cd /d "%BAYAN_DIR%"
"%VENV_PY%" -m bayansynthtts.inference %*
