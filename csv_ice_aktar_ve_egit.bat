@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "PYTHONPATH=%cd%"

echo.
echo [1/3] Python ortami kontrol ediliyor...

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
    echo Python bulunamadi. Lutfen once Python 3.11+ kur.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo [2/3] Sanal ortam olusturuluyor...
    call %PYTHON_CMD% -m venv .venv
    if errorlevel 1 (
        echo Sanal ortam olusturulamadi.
        pause
        exit /b 1
    )
)

set "PYTHON_CMD=.venv\Scripts\python.exe"
set "PIP_CMD=.venv\Scripts\pip.exe"

echo.
echo [3/3] Recursive CSV ice aktarma ve model egitimi baslatiliyor...
echo Hedef klasor: %cd%\data\bulk_import_csv

"%PYTHON_CMD%" -c "import sqlalchemy, pandas, numpy, sklearn, requests, joblib, dotenv" >nul 2>nul
if errorlevel 1 (
    echo Gerekli kutuphaneler eksik. Kurulum yapiliyor...
    call "%PYTHON_CMD%" -m pip install --upgrade pip
    call "%PIP_CMD%" install -r requirements.txt
    if errorlevel 1 (
        echo Kutuphaneler kurulurken hata olustu.
        pause
        exit /b 1
    )
)

call "%PYTHON_CMD%" scripts\import_recursive_csv_and_train.py
if errorlevel 1 (
    echo.
    echo Islem sirasinda hata olustu. Pencere acik kalacak.
    pause
    exit /b 1
)

echo.
echo Islem tamamlandi. Tum alt klasorlerdeki CSV dosyalari tarandi.
pause
endlocal
exit /b 0
