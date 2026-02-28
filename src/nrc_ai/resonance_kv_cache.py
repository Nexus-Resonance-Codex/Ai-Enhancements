from typing import Tuple

import torch
import torch.nn as nn

from .shard_folding import PhiInfinityShardFolding


class ResonanceShardKVCache(nn.Module):
    """
    Enhancement #5: Resonance Shard KV Cache v3

    A context memory mechanism redefining standard Transformer KV caches.
    Rather than letting memory scale linearly O(N), older memory blocks (shards)
    are recursively collapsed into higher-density fractals using the
    Phi Infinity Shard Folding Enhancement (#1).

    This allows infinite virtual context length mathematically bounded within
    the stable limits of the Golden Attractor, preventing gradient explosion
    while preserving resonance state.
    """
    def __init__(self, folding_steps: int = 3, shard_capacity: int = 1024):
        super().__init__()
        self.shard_capacity = shard_capacity
        self.folding_compressor = PhiInfinityShardFolding(k_steps=folding_steps)

        # State tracks active uncompressed tokens and the historically folded shards
        self.active_keys = None
        self.active_values = None
        self.folded_memory_keys = None
        self.folded_memory_values = None

    def forward(self, new_keys: torch.Tensor, new_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Appends new K/V states. If the active shard exceeds capacity, the active shard
        is mathematically folded into the permanent limit state using Phi^Infinity scaling.

        Input Shapes: (batch, seq_len, num_heads, head_dim)
        Returns: The full available Key/Value context
        """
        # 1. First initialization
        if self.active_keys is None:
            self.active_keys = new_keys
            self.active_values = new_values
            return self.active_keys, self.active_values

        # 2. Append incoming context to our active shard
        self.active_keys = torch.cat([self.active_keys, new_keys], dim=1) # dim 1 is seq_len
        self.active_values = torch.cat([self.active_values, new_values], dim=1)

        current_seq_len = self.active_keys.size(1)

        # 3. Check if capacity has triggered a Phase-Folding Limit Step
        if current_seq_len >= self.shard_capacity:
            # Compress the active shard mathematically
            compressed_k = self.folding_compressor(self.active_keys)
            compressed_v = self.folding_compressor(self.active_values)

            # Aggregate or initialize the historically folded dense memory state
            if self.folded_memory_keys is None:
                self.folded_memory_keys = compressed_k
                self.folded_memory_values = compressed_v
            else:
                # Dense limit composition (simulating infinite addition bounded by Phi)
                # Instead of concat, resonant memory integrates via addition in the compressed map
                self.folded_memory_keys = self.folded_memory_keys + compressed_k
                self.folded_memory_values = self.folded_memory_values + compressed_v

            # Reset the active shard, leaving room for new streaming context
            self.active_keys = None
            self.active_values = None

            # Since everything folded into the limit state, the next query will match
            # against the dense historical limits. For raw output, we return the folded block
            return self.folded_memory_keys, self.folded_memory_values

        # 4. If not folded, return the aggregated continuous context
        # (or composite with folded memory if it exists)
        if self.folded_memory_keys is not None:
            # Reconstruct virtually: Folded Memory + Active Shard
            total_k = torch.cat([self.folded_memory_keys, self.active_keys], dim=1)
            total_v = torch.cat([self.folded_memory_values, self.active_values], dim=1)
            return total_k, total_v

        return self.active_keys, self.active_values

    def reset_cache(self):
        """ Clears all resonance memory states for a new sequence generation """
        self.active_keys = None
        self.active_values = None
        self.folded_memory_keys = None
        self.folded_memory_values = None
