import torch
from nrc_math import PHI_FLOAT

from nrc_ai.qrt_attention_bias import QRTGeometricAttentionBias


def test_qrt_attention_bias() -> None:
    """Validates geometric damping bias using exact mathematical structural constants."""
    layer = QRTGeometricAttentionBias(max_seq_len=64)
    raw = torch.zeros(1, 1, 64, 64)
    # Forward pass: logit + cos(arctan(sqrt(phi)))
    # Note: cos(arctan(sqrt(phi))) = 1 / sqrt(1 + (sqrt(phi))^2) = 1 / sqrt(1 + phi) = 1 / phi
    biased = layer(raw)

    expected = 1.0 / PHI_FLOAT
    # Float32 precision requires 1e-5 rtol
    assert torch.allclose(biased, torch.tensor(expected, dtype=torch.float32), rtol=1e-5), (
        f"Geometric convergence breach. Found: {biased.mean().item()}, Expected: {expected}"
    )
