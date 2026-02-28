#!/usr/bin/env bash
# =================================================================
#  NRC AI Enhancements — Virtual Environment Setup Script
# =================================================================
#  Creates a local Python virtual environment and installs all
#  dependencies without modifying your system Python.
#
#  USAGE:
#    chmod +x setup_venv.sh
#    ./setup_venv.sh
#
#  PREREQUISITES:
#    - Python 3.10+ installed
#    - python3-venv package installed:
#        Pop!_OS / Ubuntu / Debian:
#          sudo apt install python3.12-venv
#        Fedora:
#          sudo dnf install python3-venv
#        Arch:
#          (included with python package)
#        macOS:
#          (included with Python from python.org or Homebrew)
#        Windows:
#          (included with Python from python.org)
# =================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"

echo "═══════════════════════════════════════════════════════════"
echo "  NRC AI Enhancement Suite — Virtual Environment Setup"
echo "═══════════════════════════════════════════════════════════"

# Create virtual environment
if [ -d "${VENV_DIR}" ]; then
    echo "[!] Virtual environment already exists at ${VENV_DIR}"
    echo "    Delete it first if you want to recreate: rm -rf ${VENV_DIR}"
    exit 1
fi

echo "[1/3] Creating virtual environment..."
python3 -m venv "${VENV_DIR}"

echo "[2/3] Activating and installing dependencies..."
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip
pip install -e ".[dev]"

echo "[3/3] Verifying installation..."
python -c "
import torch
import mpmath
from src.nrc_math.phi import PHI_FLOAT
print(f'  PyTorch: {torch.__version__}')
print(f'  φ = {PHI_FLOAT:.15f}')
print('  All imports OK!')
"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Setup complete! Activate with:"
echo "    source .venv/bin/activate"
echo ""
echo "  Then run the demo:"
echo "    python examples/demo_all_enhancements.py"
echo ""
echo "  Or run tests:"
echo "    python -m pytest tests/ -v"
echo "═══════════════════════════════════════════════════════════"
