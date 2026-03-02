import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from enhancements.tupt_sync_seed import AttractorSynchronizationSeed

def test_tupt_sync_seed():
    """
    Validates Enhancement #14: Checks that the hardware randomization pool
    is locked deterministically to the TUPT base (3697 * 2187).
    """
    # 1. Call the Attractor Seed lock
    locked_seed_val = AttractorSynchronizationSeed.synchronize()

    # 2. Verify mathematical composition
    assert locked_seed_val == (3697 * 2187), "Hardware RNG Seed initialized on an off-axis (non-resonant) boundary."

    # 3. Verify deterministic execution properties
    # Two identical tensors drawn sequentially should NO LONGER be random if the seed lock applied globally.
    AttractorSynchronizationSeed.synchronize()
    tensor_a = torch.randn(10, 10)

    AttractorSynchronizationSeed.synchronize()
    tensor_b = torch.randn(10, 10)

    assert torch.allclose(tensor_a, tensor_b), "CUDA/CPU RNG pools bypassed the Attractor Synchronization lock."
    print("Test passed: Hardware RNG pools successfully locked onto the Mod 2187 3-6-9-7 TUPT topology.")

if __name__ == "__main__":
    test_tupt_sync_seed()
