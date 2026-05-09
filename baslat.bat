@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
set "PYTHONPATH=%cd%"
set "DEFAULT_PORT=8502"
set "STREAMLIT_PORT=%DEFAULT_PORT%"
set "APP_HOST=127.0.0.1"

echo.
echo [1/4] Python ortami kontrol ediliyor...

set "PYTHON_CMD="
if exist ".venv\Scripts\python.exe" set "PYTHON_CMD=.venv\Scripts\python.exe"

if not defined PYTHON_CMD (
    where py >nul 2>nul
    if not errorlevel 1 set "PYTHON_CMD=py -3"
)

if not defined PYTHON_CMD (
    where python >nul 2>nul
    if not errorlevel 1 set "PYTHON_CMD=python"
)

if not defined PYTHON_CMD (
    if exist "%LocalAppData%\Programs\Python\Python311\python.exe" set "PYTHON_CMD=%LocalAppData%\Programs\Python\Python311\python.exe"
)

if not defined PYTHON_CMD (
    if exist "%LocalAppData%\Programs\Python\Python312\python.exe" set "PYTHON_CMD=%LocalAppData%\Programs\Python\Python312\python.exe"
)

if not defined PYTHON_CMD (
    if exist "C:\Python311\python.exe" set "PYTHON_CMD=C:\Python311\python.exe"
)

if not defined PYTHON_CMD (
    if exist "C:\Python312\python.exe" set "PYTHON_CMD=C:\Python312\python.exe"
)

if not defined PYTHON_CMD (
    echo Python bulunamadi. Lutfen Python 3.11+ kur.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo [2/4] Sanal ortam olusturuluyor...
    call %PYTHON_CMD% -m venv .venv
    if errorlevel 1 (
        echo Sanal ortam olusturulamadi.
        pause
        exit /b 1
    )
)

set "PYTHON_CMD=.venv\Scripts\python.exe"
set "PIP_CMD=.venv\Scripts\pip.exe"
set "STREAMLIT_CMD=.venv\Scripts\streamlit.exe"

set "NEED_INSTALL=0"
if not exist "%STREAMLIT_CMD%" set "NEED_INSTALL=1"

if "%NEED_INSTALL%"=="0" (
    "%PYTHON_CMD%" -c "import streamlit, sqlalchemy, pandas, numpy, sklearn, requests, joblib, dotenv" >nul 2>nul
    if errorlevel 1 set "NEED_INSTALL=1"
)

if "%NEED_INSTALL%"=="1" (
    echo.
    echo [3/4] Kutuphaneler kuruluyor...
    call "%PYTHON_CMD%" -m pip install --upgrade pip
    call "%PIP_CMD%" install -r requirements.txt
    if errorlevel 1 (
        echo Kutuphaneler kurulurken hata olustu.
        pause
        exit /b 1
    )
)

if not exist ".env" (
    echo.
    echo .env dosyasi olusturuluyor...
    copy /Y ".env.example" ".env" >nul
)

echo.
echo [4/4] Streamlit baslatiliyor...
echo Veri bootstrap'i artik panel icinden butonlarla tetiklenir.
call :find_available_port
if errorlevel 1 (
    echo Uygun bos port bulunamadi. Lutfen 8502-8512 araligini kontrol et.
    pause
    exit /b 1
)

if not "%STREAMLIT_PORT%"=="%DEFAULT_PORT%" (
    echo Port %DEFAULT_PORT% dolu. Bos port bulundu: %STREAMLIT_PORT%
)
echo Uygulama acilinca adres: http://%APP_HOST%:%STREAMLIT_PORT%
echo.
start "" "http://%APP_HOST%:%STREAMLIT_PORT%"

call "%STREAMLIT_CMD%" run app.py --server.port %STREAMLIT_PORT% --server.address %APP_HOST% --server.headless true
if errorlevel 1 (
    echo.
    echo Uygulama baslatilamadi. Hata nedeniyle pencere kapatilmayacak.
    echo Lutfen yukaridaki mesaji kontrol et.
    pause
    exit /b 1
)

endlocal
exit /b 0

:find_available_port
for %%P in (8502 8503 8504 8505 8506 8507 8508 8509 8510 8511 8512) do (
    netstat -ano | findstr ":%%P" >nul 2>nul
    if errorlevel 1 (
        set "STREAMLIT_PORT=%%P"
        exit /b 0
    )
)
exit /b 1
