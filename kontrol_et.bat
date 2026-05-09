@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
set "PYTHONPATH=%cd%"

echo.
echo [1/4] Python ortami araniyor...

set "PYTHON_EXE="
set "PYTHON_ARGS="

if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=.venv\Scripts\python.exe"

if not defined PYTHON_EXE (
    where py >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_EXE=py"
        set "PYTHON_ARGS=-3"
    )
)

if not defined PYTHON_EXE (
    where python >nul 2>nul
    if not errorlevel 1 set "PYTHON_EXE=python"
)

if not defined PYTHON_EXE (
    echo Python bulunamadi. Once Python 3.11+ kur.
    pause
    exit /b 1
)

echo Kullanilan Python: %PYTHON_EXE% %PYTHON_ARGS%

echo.
echo [2/4] Temel paket importlari kontrol ediliyor...
call "%PYTHON_EXE%" %PYTHON_ARGS% -c "import streamlit, sqlalchemy, pandas, numpy, sklearn, requests, joblib, dotenv; print('Temel import kontrolu: OK')"
if errorlevel 1 (
    echo Temel import kontrolu basarisiz.
    pause
    exit /b 1
)

echo.
echo [3/4] Ana dosyalar derleniyor...
call "%PYTHON_EXE%" %PYTHON_ARGS% -m py_compile app.py football_analytics\streamlit_app.py football_analytics\recommend.py football_analytics\db.py
if errorlevel 1 (
    echo Derleme kontrolu basarisiz.
    pause
    exit /b 1
)
echo Derleme kontrolu: OK

echo.
echo [4/4] Modul import testi yapiliyor...
call "%PYTHON_EXE%" %PYTHON_ARGS% -c "import football_analytics.streamlit_app as s; import football_analytics.recommend as r; import football_analytics.db as d; print('Modul import testi: OK')"
if errorlevel 1 (
    echo Modul import testi basarisiz.
    pause
    exit /b 1
)

echo.
echo Tum temel kontroller basariyla gecti.
pause
endlocal
exit /b 0
