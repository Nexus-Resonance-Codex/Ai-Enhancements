#  Nexus Resonance Codex - 2025-2026 Breakthrough Series
#  Copyright (c) 2026 James Trageser (@jtrag)
#
#  Licensed under CC-BY-NC-SA-4.0 + NRC-L
#  "This work is part of the Nexus Resonance Codex (NRC) incorporating TTT
#  modular exclusion, phi^inf compression, 256D->729D lattice, QRT, and MST."

"""Tests for GoldenFlowNorm module."""

import torch
from nrc_ai.golden_flow_norm import GoldenFlowNorm


def test_golden_flow_norm_initialization() -> None:
    """Verify GoldenFlowNorm initialization."""
    norm = GoldenFlowNorm(hidden_dim=128)
    # phi constant is 1.618033988749895
    assert torch.allclose(torch.tensor(norm.phi), torch.tensor(1.618033988749895))


def test_golden_flow_norm_forward() -> None:
    """Verify GoldenFlowNorm forward pass handles outliers."""
    hidden_dim = 128
    norm = GoldenFlowNorm(hidden_dim=hidden_dim)
    
    # Test shape preservation
    x = torch.randn(2, 4, hidden_dim)
    out = norm(x)
    assert out.shape == x.shape
    
    # Test outlier clamping behavior (massive explosion)
    x_exploding = torch.randn(2, 4, hidden_dim) * 1e10
    out_stabilized = norm(x_exploding)
    
    assert not torch.isnan(out_stabilized).any()
    assert not torch.isinf(out_stabilized).any()
    # Should be constrained within the geometric bounds (phi-based)
    assert torch.max(torch.abs(out_stabilized)) < 1e5  # Reasonable bound for phi-scaling
