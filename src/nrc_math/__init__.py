"""
NRC Math Library
================
Core mathematical constants and transforms underpinning all 30 enhancements:
  - phi.py:            Golden Ratio (φ) constants, Binet's formula, φ^∞ folding
  - qrt.py:            Quadratic Residue Transform for spectral filtering
  - mst.py:            Modular Stability Transform for Lyapunov-bounded clipping
  - tupt_exclusion.py: Tesla Universal Prime Transform exclusion logic
"""

from .phi import PHI_FLOAT, PHI_INVERSE_FLOAT, binet_formula, phi_power_tensor, phi_infinity_fold
from .qrt import qrt_damping
from .mst import mst_step, MST_MODULUS, MST_LAMBDA
from .tupt_exclusion import tupt_base_check, apply_exclusion_gate, TUPT_MOD, TUPT_PATTERN

__all__ = [
    "PHI_FLOAT",
    "PHI_INVERSE_FLOAT",
    "binet_formula",
    "phi_power_tensor",
    "phi_infinity_fold",
    "qrt_damping",
    "mst_step",
    "MST_MODULUS",
    "MST_LAMBDA",
    "tupt_base_check",
    "apply_exclusion_gate",
    "TUPT_MOD",
    "TUPT_PATTERN",
]
