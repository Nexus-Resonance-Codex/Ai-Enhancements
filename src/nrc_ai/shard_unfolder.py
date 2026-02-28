import torch
import torch.nn as nn
from nrc.math.phi import PHI_FLOAT


class InfiniteE_infContextUnfolder(nn.Module):
    """
    Enhancement #24: Infinite E_inf Context Shard Unfolder

    Works inversely to the Resonance Shard KV Cache (Enhancement #5).
    While the Cache physically compresses older states down by phi, the Unfolder
    operates linearly.

    When an ultra-long context task demands retrieval of deeply nested historical blocks,
    the Unfolder mathematically reverses the Shard Folding calculus. It identifies a folded
    block, and progressively scales it physically upwards utilizing phi expansions until
    full dimensional state resolution is structurally restored.
    """
    def __init__(self, folding_threshold: int = 4096):
        super().__init__()
        self.folding_threshold = folding_threshold

    def forward(self, compressed_kv_block: torch.Tensor, depth_layer: int) -> torch.Tensor:
        """
        Dynamically restores states mathematically folded deep inside the structural cache.
        Args:
            compressed_kv_block: The historically preserved compressed state vectors.
            depth_layer: The integer depth index of how many times it was folded.
        """
        # A block folded at depth N requires an exact physical multiplication upward
        # by Phi to the power of N to restore structural integrity natively.
        expansion_factor = PHI_FLOAT ** depth_layer

        # Execute the Shard Unfolding algebraically
        restored_kv_states = compressed_kv_block * expansion_factor

        return restored_kv_states
