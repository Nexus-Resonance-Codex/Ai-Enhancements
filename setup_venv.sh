#!/usr/bin/env bash
# ===========================================================
#  nrc_ai — Virtual Environment Setup
# ===========================================================
#  Creates an isolated .venv, installs the nrc core library
#  from GitHub, installs PyTorch (CPU), and then installs
#  the nrc_ai package in editable mode.
#
#  PREREQUISITE (one-time, Pop!_OS / Ubuntu / Debian):
#    sudo apt install python3.12-venv
#
#  USAGE:  ./setup_venv.sh
#          source .venv/bin/activate
#          python examples/demo_all_enhancements.py
# ===========================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'

echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  nrc_ai Library — Virtual Environment Setup${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"

# ── Prerequisite check ─────────────────────────────────────
if ! python3 -m venv --help &>/dev/null; then
    echo -e "${RED}[!] python3-venv is not available on this system.${NC}"
    echo ""
    echo "    Run this first, then re-run ./setup_venv.sh:"
    echo "      sudo apt install python3.12-venv"
    exit 1
fi

if [ -d "${VENV_DIR}" ]; then
    echo -e "${RED}[!] .venv already exists. To recreate: rm -rf .venv${NC}"
    exit 1
fi

echo "[1/5] Creating virtual environment at .venv ..."
python3 -m venv "${VENV_DIR}"

echo "[2/5] Upgrading pip ..."
"${VENV_DIR}/bin/pip" install --upgrade pip --quiet

echo "[3/5] Installing nrc core library (from GitHub) ..."
"${VENV_DIR}/bin/pip" install \
    "nrc @ git+https://github.com/Nexus-Resonance-Codex/NRC.git" --quiet

echo "[4/5] Installing PyTorch (CPU) + nrc_ai package ..."
# CPU-only torch — avoids a multi-GB CUDA install; GPU users can reinstall torch[gpu] after
"${VENV_DIR}/bin/pip" install torch --index-url https://download.pytorch.org/whl/cpu --quiet
"${VENV_DIR}/bin/pip" install -e ".[dev]" --no-deps --quiet
"${VENV_DIR}/bin/pip" install pytest ruff mypy pyyaml numpy --quiet

echo "[5/5] Verifying installation ..."
"${VENV_DIR}/bin/python" -c "
import torch, nrc
from nrc.math.phi import PHI_FLOAT
print(f'  nrc version     : {nrc.__version__}')
print(f'  torch version   : {torch.__version__}')
print(f'  φ               : {PHI_FLOAT:.6f}')
print('  ✓ All imports OK!')
"

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  ✓ Setup complete!${NC}"
echo ""
echo "  Activate with:   source .venv/bin/activate"
echo "  Full demo:       python examples/demo_all_enhancements.py"
echo "  Run tests:       pytest tests/ -v"
echo ""
echo "  GPU users: after activation, upgrade torch with:"
echo "    pip install torch --index-url https://download.pytorch.org/whl/cu121"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
