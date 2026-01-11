#!/bin/bash
# Script to run unit tests for the Breast Cancer ML Classifier project

echo "==================================="
echo "Breast Cancer ML Classifier Tests"
echo "==================================="
echo ""

# Check if virtual environment is activated
if [[ -z "${VIRTUAL_ENV}" ]]; then
    echo "⚠️  Warning: No virtual environment detected"
    echo "It's recommended to activate your virtual environment first:"
    echo "  Windows: venv\\Scripts\\activate"
    echo "  Linux/Mac: source venv/bin/activate"
    echo ""
fi

# Install testing dependencies
echo "📦 Installing testing dependencies..."
pip install -r requirements-dev.txt

echo ""
echo "🧪 Running unit tests..."
echo ""

# Run tests with coverage
pytest --cov=app/src --cov-report=term-missing --cov-report=html -v

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ All tests passed!"
    echo ""
    echo "📊 Coverage report generated at: htmlcov/index.html"
    echo "   Open it with: start htmlcov/index.html (Windows) or open htmlcov/index.html (Mac/Linux)"
else
    echo ""
    echo "❌ Some tests failed. Please review the output above."
    exit 1
fi
