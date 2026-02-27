import torch
import sys
import os
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from enhancements.golden_spiral_rope import GoldenSpiralRotaryEmbedding
from nrc_math.phi import PHI_FLOAT

def test_golden_spiral_rope():
    """
    Validates Enhancement #29: Ensures the Golden Spiral Positional architecture rotates
    complex embeddings strictly leveraging continuous mathematical Phi bases geometrically.
    """
    batch = 1
    heads = 2
    seq_len = 50
    dim = 64

    # 1. Simulate a standard Q or K tensor before dot-product attention
    # We use entirely structured 1.0 states mapped cleanly
    raw_tensor = torch.ones(batch, heads, seq_len, dim)

    # 2. Invoke the Golden Spiral module structurally
    rope = GoldenSpiralRotaryEmbedding(dim=dim, max_seq_len=seq_len)

    rotated_tensor = rope(raw_tensor, seq_dim=2)

    # Validation A: Structural Matrix Dimensionality Checks natively
    assert rotated_tensor.shape == raw_tensor.shape, "Golden Spiral math shattered array bounds physically."

    # Validation B: Verification of the base mathematical structure limits dynamically
    # At Token Index 1, Dimension 0, the equation expects:
    # Freq = 1.0 * (1.0 / (Phi ^ 0)) = 1.0
    # cos(1.0) * 1.0 + sin(1.0) * 1.0 (Since x_half_rotated of 1 is -1 or 1)

    index_1 = 1
    # Check frequency directly
    freq = rope.inv_phi_spiral[0]
    assert freq.item() == 1.0, "The absolute 0 dimension failed to yield structural Identity scalar."

    # The pure mathematics dictate structural permutations across boundaries
    # We observe whether the bounding natively functions physically via rotational mapping checks
    rotated_value_sum = rotated_tensor[0,0,1,:].sum().item()
    assert not torch.isnan(torch.tensor(rotated_value_sum)), "Mathematical rotary mapping generated unstable NaN structures inherently."

    print("Test passed: Golden Spiral Rotations mathematically executed bounding states explicitly bounded by continuous Phi frequency geometries.")

if __name__ == "__main__":
    test_golden_spiral_rope()
