#!/bin/bash
echo "=========================================="
echo "LAMBDA AI - CONTINUATION EXACT OPTIMIZATION"
echo "Using stage5_continuation_exact.py (CORRECT SCRIPT)"
echo "=========================================="

# Check Python version
echo "Python version:"
python --version

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check CUDA
echo "Checking CUDA and PyTorch..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"

# Check data structure
echo "Checking data structure..."
ls -la processed_data/splits/patient_based/

# Run the CORRECT script
echo "Starting optimization with stage5_continuation_exact.py..."
python stage5_continuation_exact.py

echo "=========================================="
echo "COMPLETED"
echo "=========================================="
