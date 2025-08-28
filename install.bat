@echo off
SETLOCAL

echo 🚀 Setting up MedSearch environment on Windows...

:: 1. Create virtual environment
IF NOT EXIST .venv (
    echo 📦 Creating Python virtual environment (.venv)
    python -m venv .venv
)

:: 2. Activate venv
CALL .venv\Scripts\activate

:: 3. Upgrade pip
echo ⬆️  Upgrading pip...
python -m pip install --upgrade pip

:: 4. Install Python dependencies
echo 📥 Installing Python requirements...
pip install -r requirements.txt

echo ✅ Installation complete!
echo.
echo Next steps:
echo 1. Activate environment: CALL .venv\Scripts\activate
echo 2. Build index: python app\build_index.py --input_dir sample_data --output_dir storage
echo 3. Search: python app\search_index.py --index_dir storage --query_text "pneumonia chest x-ray"

ENDLOCAL
pause
