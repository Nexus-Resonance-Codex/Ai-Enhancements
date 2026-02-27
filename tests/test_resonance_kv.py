import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from enhancements.resonance_kv_cache import ResonanceShardKVCache

def test_resonance_kv_cache():
    """
    Validates Enhancement #5: Resonance Shard KV Cache correctly captures incoming KV blocks,
    triggers a Phase-Folding compression at the memory boundary, and aggregates historic states
    without breaking dimensionality.
    """
    batch_size = 2
    num_heads = 12
    head_dim = 64
    shard_cap = 500  # set small for testing simulation

    kv_cache = ResonanceShardKVCache(folding_steps=2, shard_capacity=shard_cap)

    # 1. Insert an active block below capacity
    seq_block_1 = 300
    k1 = torch.randn(batch_size, seq_block_1, num_heads, head_dim)
    v1 = torch.randn(batch_size, seq_block_1, num_heads, head_dim)

    out_k1, out_v1 = kv_cache(k1, v1)

    assert out_k1.size(1) == 300, "Active cache failed to store initial sequence."
    assert kv_cache.folded_memory_keys is None, "Memory folded prematurely."

    # 2. Insert the second block pushing it PAST the memory limit
    seq_block_2 = 250
    k2 = torch.randn(batch_size, seq_block_2, num_heads, head_dim)
    v2 = torch.randn(batch_size, seq_block_2, num_heads, head_dim)

    # Combined length is 550, which > 500, triggering a folding phase immediately.
    out_k2, out_v2 = kv_cache(k2, v2)

    # Upon folding, the cache returns the mathematically compressed state representing the entire 550 elements.
    # The output size should match the shape of the folded block (550 length limit mapped to 550 fractal bounds).
    assert kv_cache.folded_memory_keys is not None, "Memory failed to trigger Phi Shard Folding."
    assert kv_cache.active_keys is None, "Active pool did not clear after phase-folding."

    assert out_k2.size() == (batch_size, 550, num_heads, head_dim), "Folded limit map lost critical dimensionality mapping."
    assert not torch.isnan(out_k2).any(), "NaN in folded KV state."

    print("Test passed: Resonance Shard KV Cache dynamically folded overlapping sequence blocks into limit bounds.")

if __name__ == "__main__":
    test_resonance_kv_cache()
