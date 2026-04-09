import torch
from nrc_math import PHI_FLOAT

from nrc_ai.giza_attention_bias import GizaSlopeAttentionBias


def test_giza_attention_bias() -> None:
    """Validates Giza-Slope bias using exact 1/phi rotational exactness."""
    layer = GizaSlopeAttentionBias(max_seq_len=64)
    raw = torch.zeros(1, 1, 64, 64)
    # Forward pass: logit + 1/phi
    biased = layer(raw)

    expected = 1.0 / PHI_FLOAT
    # Float32 precision requires 1e-5 rtol
    assert torch.allclose(biased, torch.tensor(expected, dtype=torch.float32), rtol=1e-5), (
        f"Giza exactness breach. Found: {biased.mean().item()}, Expected: {expected}"
    )
