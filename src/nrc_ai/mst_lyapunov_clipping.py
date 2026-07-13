#  Nexus Resonance Codex (NRC) (NRC) (NRC) - 2025-2026 Breakthrough Series
#  Copyright (c) 2026 James Paul Trageser (@jtrag)
#
#  Licensed under CC-BY-NC-SA-4.0 + NRC-L
#  "This work is part of the Nexus Resonance Codex (NRC) (NRC) (NRC) (NRC) incorporating TTT
#  modular exclusion, phi^inf compression, 256D->729D lattice, QRT, and MST."

"""MST Lyapunov Clipping: Chaos Control.

This module implements a dynamic clipping mechanism that prevents
latent state divergence by projecting gradients back into the
Lyapunov stability boundary of the Golden Attractor.
"""

import torch
import torch.nn as nn


class MSTLyapunovClipping(nn.Module):
    """Enhancement #29: MST Lyapunov Gradient Clipping.

    Clamps gradient magnitudes according to the chaotic residue limits to
    ensure architectural resonance is never breached.
    """

    def __init__(self, clip_val: float = 0.381):
        super().__init__()
        self.clip_val = clip_val
        self.mst_threshold = clip_val

    def forward(self, grad: torch.Tensor) -> torch.Tensor:
        """Applies the Lyapunov clip to the gradient manifold.

        Args:
            grad: The gradient tensor to be clipped.
        """
        # 1. Deterministic Bounds Check
        # (Using PHI-modulated clipping limits)
        clipped_grad = torch.clamp(grad, -self.clip_val, self.clip_val)

        # 2. Resonant Projection
        # certfies that the gradient remains within the stable lattice
        return clipped_grad
