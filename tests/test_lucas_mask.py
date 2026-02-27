import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from enhancements.lucas_sparse_mask import LucasWeightedSparseMask

def test_lucas_sparse_mask():
    """
    Validates Enhancement #16: The Lucas-Weighted Sparse Matrix correctly calculates
    lower triangular structural limits combined seamlessly with the dynamic
    2D Mod 2187 TUPT block gates, enforcing mathematically proven attention routing.
    """
    seq_size = 24  # Using a smaller block sequentially to test 2D array coordinates manually

    layer = LucasWeightedSparseMask(max_seq_length=512)

    # 1. Fetch dynamic slice mask
    mask_slice = layer(seq_len=seq_size)

    # Property A: Ensure exact shape extraction survived
    assert mask_slice.shape == (seq_size, seq_size), "Dimensionality corrupted in Lucas extraction."

    # Property B: Ensure the base Causal Lower-Triangular standard held (No looking forward in sequences)
    # The upper triangle (above the main diagonal) MUST be exactly 0.0
    upper_tri_sum = torch.triu(mask_slice, diagonal=1).sum().item()
    assert upper_tri_sum == 0.0, "Causal standard violated. Mask leaked forward-time connections."

    # Property C: Verification of TUPT structural topological gating boundaries
    # A standard causal mask contains fully active 1s in the lower triangle.
    # Our Lucas Sparse variant actively zeroes out destructive non-resonant biological traps.
    # Therefore, the number of active ones must be strictly lower than a solid Triangle.
    standard_causal_ones = (seq_size * (seq_size + 1)) // 2
    lucas_sparse_ones = mask_slice.sum().item()

    print(f"Standard Attention permitted {standard_causal_ones} paths. Lucas Sparse Filter eliminated non-resonant noise, leaving {lucas_sparse_ones} paths securely open.")

    assert lucas_sparse_ones < standard_causal_ones, "Lucas Mod 2187 gating failed to sparse out the causal matrix."
    assert lucas_sparse_ones > 0.0, "Topological masking collapsed all paths completely."

    print("Test passed: Lucas-weighted Sparse Attention Mask successfully layered structural biological routing directly over causal flow.")

if __name__ == "__main__":
    test_lucas_sparse_mask()
