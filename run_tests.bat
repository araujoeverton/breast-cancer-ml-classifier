@echo off
REM Script to run unit tests for the Breast Cancer ML Classifier project (Windows)

echo ===================================
echo Breast Cancer ML Classifier Tests
echo ===================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo Warning: Virtual environment not found at venv\
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install testing dependencies
echo.
echo Installing testing dependencies...
pip install -r requirements-dev.txt

REM Run tests with coverage
echo.
echo Running unit tests...
echo.
pytest --cov=app/src --cov-report=term-missing --cov-report=html -v

REM Check exit code
if %ERRORLEVEL% EQU 0 (
    echo.
    echo All tests passed!
    echo.
    echo Coverage report generated at: htmlcov\index.html
    echo Opening coverage report...
    start htmlcov\index.html
) else (
    echo.
    echo Some tests failed. Please review the output above.
    exit /b 1
)
