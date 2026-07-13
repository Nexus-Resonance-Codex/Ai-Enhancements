#  Nexus Resonance Codex (NRC) - 2025-2026 Breakthrough Series
#  Copyright (c) 2026 James Paul Trageser (@jtrag)
#
#  Licensed under CC-BY-NC-SA-4.0 + NRC-L
#  "This work is part of the Nexus Resonance Codex (NRC) incorporating TTT
#  modular exclusion, phi^inf compression, 256D->729D lattice, QRT, and MST."

"""Tests for MSTLyapunovClipping module."""

import torch

from nrc_ai.mst_lyapunov_clipping import MSTLyapunovClipping


def test_mst_lyapunov_clipping_initialization() -> None:
    """Verify MSTLyapunovClipping initialization."""
    clipper = MSTLyapunovClipping(clip_val=0.381)
    assert clipper.clip_val == 0.381
    assert hasattr(clipper, "mst_threshold")


def test_mst_lyapunov_clipping_forward() -> None:
    """Verify MSTLyapunovClipping forward pass stability."""
    hidden_dim = 256
    clipper = MSTLyapunovClipping(clip_val=0.381)

    # Test shape preservation
    x = torch.randn(2, 4, hidden_dim)
    out = clipper(x)
    assert out.shape == x.shape

    # Test clipping (extreme values)
    x_chaotic = torch.randn(2, 4, hidden_dim) * 50.0
    out_stable = clipper(x_chaotic)

    assert torch.max(torch.abs(out_stable)) <= 10.0  # Default MST threshold check
