#!/bin/bash
set -e
echo "🔧 Setting up Diabetes Prediction API..."
pip install --upgrade pip setuptools wheel
echo "📦 Installing dependencies..."
pip install -r requirements.txt
mkdir -p uploads models logs
echo "✅ Setup complete!"
echo "🚀 Starting application..."
python app.py
