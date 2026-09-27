@echo off
setlocal EnableExtensions

cd /d "%~dp0"
set "PROJECT_DIR=%~dp0"
set "VENV_PYTHON=%~dp0.venv\Scripts\python.exe"
set "PYTHON_LAUNCHER="
set "PYTHON_EXE="

echo [1/5] Checking Python 3.11...
where py >nul 2>&1
if not errorlevel 1 py -3.11 --version >nul 2>&1
if not errorlevel 1 set "PYTHON_LAUNCHER=py"
if defined PYTHON_LAUNCHER goto :create_environment
goto :install_python

:install_python
echo Python 3.11 was not found. Trying to install it with winget...
where winget >nul 2>&1
if errorlevel 1 goto :python_error
winget install --id Python.Python.3.11 -e --scope user --accept-source-agreements --accept-package-agreements
if errorlevel 1 goto :python_error

echo.
echo Python was installed. Checking the Python launcher again...
where py >nul 2>&1
if not errorlevel 1 py -3.11 --version >nul 2>&1
if not errorlevel 1 set "PYTHON_LAUNCHER=py"
if defined PYTHON_LAUNCHER goto :create_environment

rem PATH is not always refreshed in the current cmd.exe after winget finishes.
for %%P in (
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    "%ProgramFiles%\Python311\python.exe"
    "%ProgramFiles(x86)%\Python311\python.exe"
) do if not defined PYTHON_EXE if exist "%%~P" set "PYTHON_EXE=%%~P"
if not defined PYTHON_EXE goto :python_error

:create_environment
if exist "%VENV_PYTHON%" goto :install_dependencies

echo [2/5] Creating virtual environment .venv...
if defined PYTHON_LAUNCHER (
    py -3.11 -m venv "%PROJECT_DIR%.venv"
) else (
    "%PYTHON_EXE%" -m venv "%PROJECT_DIR%.venv"
)
if errorlevel 1 goto :setup_error

:install_dependencies
if not exist "%VENV_PYTHON%" goto :setup_error

echo [3/5] Updating pip...
"%VENV_PYTHON%" -m pip install --upgrade pip
if errorlevel 1 goto :setup_error

echo [4/5] Installing Sprite Soul and its libraries...
"%VENV_PYTHON%" -m pip install -e "%PROJECT_DIR%"
if errorlevel 1 goto :setup_error

echo.
echo [5/5] Preparing AI: checking NVIDIA CUDA and downloading models...
echo This can take a while on the first run. Progress is shown below.
echo The CUDA Toolkit is not required; Sprite Soul installs the compatible PyTorch build.
"%VENV_PYTHON%" -m smg setup --models all --progress text
if errorlevel 1 goto :ai_setup_error

echo.
echo Environment and AI models are ready.
echo Start the application with run_sprite_soul.bat
pause
exit /b 0

:python_error
echo.
echo Could not install or find Python 3.11.
echo Install Python 3.11 manually from https://www.python.org/downloads/windows/
echo Then run this file again.
pause
exit /b 1

:setup_error
echo.
echo Environment setup failed. Check the error above and run this file again.
pause
exit /b 1

:ai_setup_error
echo.
echo Sprite Soul was installed, but AI preparation did not finish.
echo Check the NVIDIA driver and internet connection, then run this command again:
echo   .venv\Scripts\python.exe -m smg setup --models all --progress text
echo You can still start the application with run_sprite_soul.bat.
pause
exit /b 1
