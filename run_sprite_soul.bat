@echo off
setlocal

cd /d "%~dp0"
set "PYTHONW=%~dp0.venv\Scripts\pythonw.exe"

if not exist "%PYTHONW%" (
    echo Virtual environment was not found: .venv
    echo Create it and install the project dependencies first:
    echo   py -3.11 -m venv .venv
    echo   .venv\Scripts\python.exe -m pip install -e .
    pause
    exit /b 1
)

start "Sprite Soul" "%PYTHONW%" "%~dp0app.py"
exit /b 0
