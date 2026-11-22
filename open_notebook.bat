@echo off
setlocal
title Crop Yield Prediction - Notebook
cd /d "%~dp0"

rem ---- find Python 3.12 or newer ----
set "PY="
where py >nul 2>nul && py -3 -c "import sys; sys.exit(sys.version_info < (3, 12))" >nul 2>nul && set "PY=py -3"
if not defined PY (
    where python >nul 2>nul && python -c "import sys; sys.exit(sys.version_info < (3, 12))" >nul 2>nul && set "PY=python"
)
if not defined PY goto :nopython

rem ---- private environment, created once ----
if not exist ".venv\Scripts\python.exe" (
    echo Setting up a private Python environment for this app - first run only...
    %PY% -m venv .venv
    if errorlevel 1 goto :fail
)
echo Installing / checking packages - the first run takes a few minutes...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -q -r requirements-notebook.txt
if errorlevel 1 goto :fail

echo.
echo Opening the notebook in Jupyter. All results are already shown;
echo use Run - Run All Cells to re-run it - takes 5-10 minutes.
echo.
".venv\Scripts\python.exe" -m notebook Crops_Prediction.ipynb
pause
exit /b 0

:nopython
echo.
echo Python 3.12 or newer was not found.
echo Install it from https://www.python.org/downloads/ and tick "Add python.exe to PATH",
echo then double-click this file again.
pause
exit /b 1

:fail
echo.
echo Setup failed. Check your internet connection and try again.
echo If it keeps failing, delete the ".venv" folder here and run this file again.
pause
exit /b 1
