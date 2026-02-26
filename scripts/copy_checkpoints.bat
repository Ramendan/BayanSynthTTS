@echo off
:: BayanSynthTTS — Copy LoRA checkpoints from training workspace
::
:: Copies the best checkpoints from bayansynth/exp/ into BayanSynthTTS/checkpoints/
:: Run this after training to make them available to BayanSynthTTS.

setlocal
set "REPO_ROOT=%~dp0..\.."
set "SRC_LLM=%REPO_ROOT%\bayansynth\exp\llm_improved\epoch_28_whole.pt"
set "SRC_FLOW=%REPO_ROOT%\bayansynth\exp\flow\epoch_15_step_49000.pt"
set "DST_LLM=%~dp0..\checkpoints\llm"
set "DST_FLOW=%~dp0..\checkpoints\flow"

echo.
echo === Copying LoRA checkpoints to BayanSynthTTS/checkpoints/ ===
echo.

:: LLM LoRA
if exist "%SRC_LLM%" (
    if not exist "%DST_LLM%" mkdir "%DST_LLM%"
    copy /Y "%SRC_LLM%" "%DST_LLM%\" >nul
    echo [OK] LLM LoRA  → %DST_LLM%\epoch_28_whole.pt
) else (
    echo [--] LLM LoRA not found at: %SRC_LLM%
    echo      Train first or adjust the path in this script.
)

:: Flow LoRA
if exist "%SRC_FLOW%" (
    if not exist "%DST_FLOW%" mkdir "%DST_FLOW%"
    copy /Y "%SRC_FLOW%" "%DST_FLOW%\" >nul
    echo [OK] Flow LoRA → %DST_FLOW%\epoch_15_step_49000.pt
) else (
    echo [--] Flow LoRA not found at: %SRC_FLOW%
    echo      This is optional — flow LoRA is disabled by default.
)

echo.
echo Done.  Edit BayanSynthTTS\conf\models.yaml to point to different checkpoints.
echo.
