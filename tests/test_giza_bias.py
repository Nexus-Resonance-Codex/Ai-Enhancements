import torch
import sys
import os
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from enhancements.giza_attention_bias import GizaAngleAttentionBias

def test_giza_attention_bias():
    """
    Validates Enhancement #27: The Giza Angle matrix calculates the 51.85-degree
    cosine structure and maps it flawlessly into sequence limits.
    """
    batch = 2
    heads = 4
    seq = 64

    # 1. Simulate raw attention output matrices (Normally Q @ K.T)
    # Using entirely zeros conceptually to transparently witness the bias application
    raw_attention = torch.zeros(batch, heads, seq, seq)

    layer = GizaAngleAttentionBias(max_seq_len=seq)

    # 2. Project the architectural bias
    biased_attention = layer(raw_attention)

    # Validation A: Proper mathematical extraction
    # 51.85 degrees natively mapped to radians
    radians = 51.85 * (math.pi / 180.0)
    expected_limit = math.cos(radians)

    # Validation B: Since inputs were zeroes, the final biased array should globally equal expected_limit
    assert torch.allclose(biased_attention, torch.tensor(expected_limit), rtol=1e-4), "Giza Bias Projection failed to globally broadcast 51.85-degree limits."

    # Validation C: Structural broadcast shape physics
    assert biased_attention.shape == raw_attention.shape, "Giza Bias fractured attention head bounding."

    print(f"Test passed: Giza-Slope Attention Bias mapped the physical Cosine parameter ({expected_limit:.4f}) structurally over all dot-product sequence routes.")

if __name__ == "__main__":
    test_giza_attention_bias()
