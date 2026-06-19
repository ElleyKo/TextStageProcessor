@echo off
setlocal
cd /d "%~dp0"

set "VENV_PYTHON=%USERPROFILE%\venvs\TextStageProcessor\.venv\Scripts\python.exe"
if exist "%VENV_PYTHON%" (
    "%VENV_PYTHON%" stage_text_processor.py
) else (
    echo Virtual environment was not found: %VENV_PYTHON%
    echo Run setup_env.bat first.
)

pause
