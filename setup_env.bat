@echo off
setlocal
cd /d "%~dp0"

set "VENV_DIR=%USERPROFILE%\venvs\TextStageProcessor\.venv"
set "PYTHON=%VENV_DIR%\Scripts\python.exe"

python -m venv "%VENV_DIR%"
"%PYTHON%" -m pip install --upgrade pip "setuptools<81" wheel
"%PYTHON%" -m pip install docopt==0.6.2
"%PYTHON%" -m pip install --only-binary=:all: -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements-py313.txt

pause
