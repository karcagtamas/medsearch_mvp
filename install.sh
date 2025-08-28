#!/bin/bash
set -e

echo "🚀 Setting up MedSearch environment..."

# 1. Create virtual environment
if [ ! -d ".venv" ]; then
  echo "📦 Creating Python virtual environment (.venv)"
  python3 -m venv .venv
fi

# 2. Activate venv
source .venv/bin/activate

# 3. Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# 4. Install system dependencies
echo "🔧 Installing system dependencies (Tesseract + Poppler)..."
sudo apt-get update
sudo apt-get install -y tesseract-ocr poppler-utils

# 5. Install Python dependencies
echo "📥 Installing Python requirements..."
pip install -r requirements.txt

echo "✅ Installation complete!"
echo ""
echo "Next steps:"
echo "1. Activate environment: source .venv/bin/activate"
echo "2. Build index: python app/build_index.py --input_dir sample_data --output_dir storage"
echo "3. Search: python app/search_index.py --index_dir storage --query_text 'pneumonia chest x-ray'"
