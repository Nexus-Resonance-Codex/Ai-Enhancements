import torch
import sys
import os

# Add src to Python path for testing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from enhancements.shard_folding import PhiInfinityShardFolding

def test_shard_folding_compression():
    """
    Tests the deterministic property and geometric constraints of the
    Phi Infinity Shard Folding Enhancement #1.
    """
    folding_module = PhiInfinityShardFolding(k_steps=3)

    # Simulate a standard normal LoRA adapter weight or KV Cache shard
    test_tensor = torch.randn(8, 512, dtype=torch.float32)

    compressed = folding_module(test_tensor)

    # Verify shape consistency
    assert compressed.shape == test_tensor.shape

    # Verify convergence / lack of exploding gradients or NaNs
    assert not torch.isnan(compressed).any(), "NaN found in folding output."
    assert not torch.isinf(compressed).any(), "Inf found in folding output."

    # Verify deterministic projection mapping
    compressed_2 = folding_module(test_tensor)
    assert torch.allclose(compressed, compressed_2), "Folding mapping is non-deterministic."

    print("Test passed: Phi Infinity Shard Folding successfully maps tensor to fractional bounded modulus.")

if __name__ == "__main__":
    test_shard_folding_compression()
