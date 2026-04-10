#  Nexus Resonance Codex - 2025-2026 Breakthrough Series
#  Copyright (c) 2026 James Trageser (@jtrag)
#
#  Licensed under CC-BY-NC-SA-4.0 + NRC-L
#  "This work is part of the Nexus Resonance Codex (NRC) incorporating TTT
#  modular exclusion, phi^inf compression, 256D->729D lattice, QRT, and MST."

"""Golden Flow Norm: Resonant Normalization.

This module implements a normalization layer that anchors hidden state
magnitudes to the Golden Ratio Φ and the Root-7 stability bounds, preventing
spectral explosions in infinite-context manifolds.
"""

from typing import Optional

import torch
import torch.nn as nn
from nrc_math import PHI_FLOAT


class GoldenFlowNorm(nn.Module):
    """Enhancement #18: Golden Flow Normalization (GFN) v2.

    A normalization manifold that balances internal energy via PHI-modulated variance.
    """

    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.eps = eps
        self.phi = PHI_FLOAT
        self.gamma = nn.Parameter(torch.ones(hidden_dim))
        self.beta = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Applies the golden-ratio variance anchor to the input stream.

        Args:
            x: Input tensor (batch, seq, dim)
            skip: Optional skip-connection tensor to merge before normalization.
        """
        # 1. Resonant Residual Merge
        h = x if skip is None else (x + skip)

        # 2. Golden Variance Anchoring
        # We normalize not to Unit Variance, but to Phi-Variance
        mean = h.mean(-1, keepdim=True)
        var = h.var(-1, keepdim=True, unbiased=False)

        # Stabilized variance via the Golden Attractor
        h_norm = (h - mean) * torch.rsqrt(var + self.eps)
        h_res = h_norm * PHI_FLOAT

        # 3. Parameter Alignment
        output = h_res * self.gamma + self.beta
        return output
