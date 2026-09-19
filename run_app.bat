@echo off
chcp 65001 >nul
title Mizan AI - المستشار القانوني الليبي الذكي

echo ============================================
echo   Mizan AI - المستشار القانوني الليبي الذكي
echo ============================================
echo.

cd /d "%~dp0"

REM ─── Create virtual environment on first run ───
if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: Python not found. Please install Python 3.10+ from python.org
        pause
        exit /b 1
    )
)

REM ─── Install / update requirements ───
echo [2/3] Installing requirements ^(first run only^)...
".venv\Scripts\python.exe" -m pip install --quiet --upgrade pip
".venv\Scripts\python.exe" -m pip install --quiet -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install requirements. Check your internet connection.
    pause
    exit /b 1
)

REM ─── Launch Streamlit ───
echo [3/3] Starting Mizan AI...
echo.
".venv\Scripts\python.exe" -m streamlit run app.py

pause
