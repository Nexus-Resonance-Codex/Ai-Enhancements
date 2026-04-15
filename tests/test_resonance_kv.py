import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from nrc_ai.resonance_kv_cache import ResonanceShardKVCache


def test_resonance_kv_cache() -> None:
    """Validates Enhancement #5: Resonance Shard KV Cache correctly captures incoming KV blocks,
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
    # Note: PhiInfinityShardFolding preserves shape.
    out_k2, out_v2 = kv_cache(k2, v2)

    # Upon folding, the cache returns the mathematically compressed state representing the entire 550 elements.
    assert kv_cache.folded_memory_keys is not None, "Memory failed to trigger Phi Shard Folding."
    assert kv_cache.active_keys is None, "Active pool did not clear after phase-folding."

    # ShardFolding preserves sequence length
    assert out_k2.size(1) == 550, f"Folded limit map lost critical dimensionality. Got {out_k2.size(1)}"
    assert not torch.isnan(out_k2).any(), "NaN in folded KV state."

    # 3. Insert a third block to verify accumulation (Active Keys Initialization Branch)
    seq_block_3 = 100
    k3 = torch.randn(batch_size, seq_block_3, num_heads, head_dim)
    v3 = torch.randn(batch_size, seq_block_3, num_heads, head_dim)

    # self.active_keys is None, so step 1 returns k3. size=100.
    out_k3, out_v3 = kv_cache(k3, v3)
    assert out_k3.size(1) == 100
    assert kv_cache.active_keys is not None
    assert kv_cache.folded_memory_keys is not None

    # 4. Trigger Step 4 Branch (Folded Memory + Active Shard Reconstruction)
    # We have folded_memory(550) and active_keys(100).
    # If we add a SMALL block (e.g. 50), total length 100+50=150 < 500.
    # Code proceeds to Step 4 and cats folded_memory(550) + active(150) = 700.
    seq_block_4 = 50
    k4 = torch.randn(batch_size, seq_block_4, num_heads, head_dim)
    v4 = torch.randn(batch_size, seq_block_4, num_heads, head_dim)
    out_k4, _ = kv_cache(k4, v4)
    assert out_k4.size(1) == 700, f"Expected 700, got {out_k4.size(1)}"

    # 5. Trigger a SECOND fold (Step 3 branch with historical memory)
    # Current active is 150. Add 400 to reach 550 > 500.
    seq_block_5 = 400
    k5 = torch.randn(batch_size, seq_block_5, num_heads, head_dim)
    v5 = torch.randn(batch_size, seq_block_5, num_heads, head_dim)
    out_k5, _ = kv_cache(k5, v5)
    # The new compressed shard(550) length matches old folded(550), so they add.
    # Length remains 550.
    assert out_k5.size(1) == 550, f"Expected 550, got {out_k5.size(1)}"

    # 6. Test sequence length mismatch handling (Step 1 Reset Branch inside Forward)
    # Setting active_keys to None and changing capacity triggers a fresh shard setup.
    kv_cache.active_keys = None  # Force Step 1
    seq_block_6 = 850
    k6 = torch.randn(batch_size, seq_block_6, num_heads, head_dim)
    v6 = torch.randn(batch_size, seq_block_6, num_heads, head_dim)
    out_k6, _ = kv_cache(k6, v6)
    assert out_k6.size(1) == 850

    # Now push past capacity (current shard_capacity is 500)
    # current active is 850 >= 500. Folding triggers NEXT call.
    # Wait, Step 2 cats, then Step 3 checks.
    # To trigger the mismatch addition branch (85 != 81), we need to fold an 850-length shard.
    # The next call with any size will trigger Step 2 (cat) then Step 3 (fold).
    seq_block_7 = 10
    k7 = torch.randn(batch_size, seq_block_7, num_heads, head_dim)
    v7 = torch.randn(batch_size, seq_block_7, num_heads, head_dim)
    out_k7, _ = kv_cache(k7, v7)
    # active becomes 860. Folded becomes compressed(860). size=860.
    assert out_k7.size(1) == 860
    assert kv_cache.folded_memory_keys.size(1) == 860

    # 7. Test cache reset (Step 5 branch)
    kv_cache.reset_cache()
    assert kv_cache.active_keys is None
    assert kv_cache.folded_memory_keys is None

    print("Test passed: Resonance Shard KV Cache accumulated and folded blocks correctly with full coverage.")


if __name__ == "__main__":
    test_resonance_kv_cache()
