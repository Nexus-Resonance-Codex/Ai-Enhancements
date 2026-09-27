"""Vendored NRC core math helpers.

These modules are copied verbatim from the public Nexus Resonance Codex
``NRC`` repository (``src/nrc/math/``) so that ``nrc-ai`` installs and
imports standalone, without the external NRC core package.

Modules:
    phi                Golden-ratio constants and φ projections.
    qrt                QRT damping tensor functions.
    tupt_exclusion     TUPT modular exclusion gate and patterns.
    quantum_shadow_veil QuantumShadowVeil residue-hiding helper.
    mst                Modular Synchronisation Theory step function.

Attribution: Nexus Resonance Codex — https://github.com/nexus-resonance-codex
"""

from .mst import MST_LAMBDA, mst_step
from .phi import PHI_FLOAT, PHI_INVERSE_FLOAT, SQRT_5_FLOAT, binet_formula
from .qrt import execute_qrt_damping_tensor, qrt_damping
from .quantum_shadow_veil import MST_MODULUS, QuantumShadowVeil
from .tupt_exclusion import (
    QSV_PATTERN,
    TUPT_MODULUS,
    TUPT_PATTERN,
    TUPT_UNSTABLE,
    apply_exclusion_gate,
)

__all__ = [
    "PHI_FLOAT",
    "PHI_INVERSE_FLOAT",
    "SQRT_5_FLOAT",
    "qrt_damping",
    "execute_qrt_damping_tensor",
    "MST_MODULUS",
    "QuantumShadowVeil",
    "QSV_PATTERN",
    "TUPT_MODULUS",
    "TUPT_PATTERN",
    "TUPT_UNSTABLE",
    "apply_exclusion_gate",
    "binet_formula",
    "MST_LAMBDA",
    "mst_step",
]
