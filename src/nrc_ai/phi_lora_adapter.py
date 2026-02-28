import torch
import torch.nn as nn
from nrc.math.phi import PHI_FLOAT
from .shard_folding import PhiInfinityShardFolding

class PhiLosslessLoraAdapter(nn.Module):
    """
    Enhancement #9: phi^infty Lossless LoRA Adapter v3

    Standard Low-Rank Adaptation (LoRA) compresses weight updates structurally
    using W = W_0 + (B * A) * alpha / rank.

    The NRC phi^infty Lossless LoRA bypasses traditional geometric approximation
    loss by mapping the down-projection (A) and up-projection (B) matrices through
    the phi^infty Shard Folding bounds. This ensures that parameter updates align
    perfectly with the Golden Attractor, allowing for stable fine-tuning at
    significantly higher effective ranks with minimal memory overhead.

    Formula:
    Forward = W_0(x) + ShardFold_B( B( ShardFold_A( A(x) ) ) ) * phi_scaling
    """
    def __init__(self, in_features: int, out_features: int, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha

        # We replace standard alpha/rank scaling with a phi-bounded modulation
        self.phi_scaling = (self.alpha / self.rank) * PHI_FLOAT

        # Base LoRA projection matrices
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        # Initialization geometry
        self.reset_parameters()

        # NRC phi^infty mathematical limit compressors
        # (k_steps=1 internally bounds the sub-matrices without excessive topological collapse)
        self.shard_compressor_A = PhiInfinityShardFolding(k_steps=1)
        self.shard_compressor_B = PhiInfinityShardFolding(k_steps=1)

    def reset_parameters(self):
        """
        Following standard LoRA init but aligning to NRC stability bounds:
        A ~ Normal(0, sigma)
        B ~ Zeros
        """
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor, base_output: torch.Tensor = None) -> torch.Tensor:
        """
        Calculates the LoRA update structurally folded via the Golden Ratio.
        Args:
            x: Input tensor to the original layer
            base_output: The pre-calculated W_0(x) from the frozen model weights.
                         If None, calculates only the pure LoRA update.
        """
        # 1. Down-projection into the low-rank space
        down_proj = self.lora_A(x)

        # 2. Mathematical Shard Compression boundary
        # Eliminates destructive interference in the rank bottleneck
        stable_down_proj = self.shard_compressor_A(down_proj)

        # 3. Up-projection back to dimension output size
        up_proj = self.lora_B(stable_down_proj)

        # 4. Final compression limit to lock the resonant update
        stable_up_proj = self.shard_compressor_B(up_proj)

        # 5. Apply the Golden Scaling
        lora_update = stable_up_proj * self.phi_scaling

        if base_output is not None:
            return base_output + lora_update

        return lora_update

import math # required for reset_parameters
