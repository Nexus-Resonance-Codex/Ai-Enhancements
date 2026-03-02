import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from enhancements.shard_unfolder import InfiniteE_infContextUnfolder
from nrc_math.phi import PHI_FLOAT

def test_infinite_e_inf_unfolder():
    """
    Validates Enhancement #24: Ensures the Context Unfolder structurally reverses
    Phi-Shard memory compression algorithms, expanding cached geometries back to
    native execution size linearly.
    """
    batch = 1
    seq = 10
    dim = 64
    depth_layer = 4 # Imagine a memory block folded 4 times deep into the Phi cache

    # 1. Create a simulated memory block originally "compressed" down by phi^4
    original_math_state = torch.ones(batch, seq, dim)
    compressed_state = original_math_state * (1.0 / (PHI_FLOAT ** depth_layer))

    # 2. Deploy the Algebraic Unfolder
    unfolder = InfiniteE_infContextUnfolder()

    restored_state = unfolder(compressed_state, depth_layer=depth_layer)

    # Validation A: Structural footprint check
    assert restored_state.shape == compressed_state.shape, "Unfolding grid corrupted tensor depth matrices natively."

    # Validation B: Verification of lossless Algebraic Unfolding
    # We must recover exactly the original logical state prior to Phi-shard compression.
    assert torch.allclose(restored_state, original_math_state, rtol=1e-4), "Phi expansion physics failed to losslessly recover compressed structural bounds."

    print(f"Test passed: Infinite E_inf Unfolder seamlessly restored deep memory contexts algebraically (Depth: {depth_layer} Layers).")

if __name__ == "__main__":
    test_infinite_e_inf_unfolder()
