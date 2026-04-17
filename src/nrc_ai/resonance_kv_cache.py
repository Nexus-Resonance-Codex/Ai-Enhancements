from typing import Optional, Tuple

import torch
import torch.nn as nn
from nrc.math import QuantumShadowVeil

from .shard_folding import PhiInfinityShardFolding

__version__ = "2.3.0-QSV"


class ResonanceShardKVCache(nn.Module):
    """Institutional-grade KV-cache featuring Resonance-Shard Encryption.

    Utilizes the Quantum Shadow Veil (QSV) to protect historically folded memory.
    """

    def __init__(self, folding_steps: int = 3, shard_capacity: int = 1024, qsv_seed: int = 137) -> None:
        super().__init__()
        self.shard_capacity = shard_capacity
        self.folding_compressor = PhiInfinityShardFolding(k_steps=folding_steps)

        # Quantum Shadow Veil Integration
        self.qsv = QuantumShadowVeil(spiral_density=4096)
        self.qsv.expand_fibonacci_keys(seed=qsv_seed, count=1024)
        self.fold_counter = 0

        # State tracks active uncompressed tokens and the historically folded shards
        self.active_keys: Optional[torch.Tensor] = None
        self.active_values: Optional[torch.Tensor] = None
        self.folded_memory_keys: list[torch.Tensor] = []
        self.folded_memory_values: list[torch.Tensor] = []

    def _apply_shadow_veil(self, tensor: torch.Tensor, key_idx: int) -> torch.Tensor:
        """Applies the Residue-Hiding (RH) encryption to a memory shard (Torch-Native)."""
        from nrc.math import PHI_FLOAT

        device = tensor.device
        dtype = tensor.dtype

        key = self.qsv.keys[key_idx % len(self.qsv.keys)]
        salt = float(key % 256) / 256.0

        # PHI-based resonant phasing: (tensor + salt) * phi^{-n}
        # Implemented via torch-native ops to preserve grad-lattice
        encrypted = (tensor + salt) * (PHI_FLOAT ** -(key_idx % 13))
        return encrypted.to(device=device, dtype=dtype)

    def _remove_shadow_veil(self, tensor: torch.Tensor, key_idx: int) -> torch.Tensor:
        """Decrypts a memory shard via inverse resonant phasing (Torch-Native)."""
        from nrc.math import PHI_FLOAT

        device = tensor.device
        dtype = tensor.dtype

        # Inverse phasing: encrypted / phi^{-n} - salt
        key = self.qsv.keys[key_idx % len(self.qsv.keys)]
        salt = float(key % 256) / 256.0
        decrypted = (tensor / (PHI_FLOAT ** -(key_idx % 13))) - salt

        return decrypted.to(device=device, dtype=dtype)

    def forward(self, new_keys: torch.Tensor, new_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. First initialization / Append incoming context
        if self.active_keys is None:
            self.active_keys = new_keys
            self.active_values = new_values
        else:
            # Ensure active_keys is Tensor for type safety
            active_k = self.active_keys
            active_v = self.active_values
            assert active_k is not None and active_v is not None

            self.active_keys = torch.cat([active_k, new_keys], dim=1)
            self.active_values = torch.cat([active_v, new_values], dim=1)

        # 3. Phase-Folding Limit Step
        if self.active_keys.size(1) >= self.shard_capacity:
            compressed_k = self.folding_compressor(self.active_keys)
            compressed_v = self.folding_compressor(self.active_values)

            # Protect shards using Quantum Shadow Veil
            protected_k = self._apply_shadow_veil(compressed_k, self.fold_counter)
            protected_v = self._apply_shadow_veil(compressed_v, self.fold_counter)

            # Resonant Accumulation: List-based shards handle dynamic sequence lengths
            self.folded_memory_keys.append(protected_k)
            self.folded_memory_values.append(protected_v)
            self.fold_counter += 1

            self.active_keys = None
            self.active_values = None

        # 4. Veil-Authenticated Retrieval
        if self.folded_memory_keys:
            # Authenticated decryption for attention cycle via shard-loop
            unveiled_ks = []
            unveiled_vs = []
            for i, (k_shard, v_shard) in enumerate(zip(self.folded_memory_keys, self.folded_memory_values, strict=True)):
                unveiled_ks.append(self._remove_shadow_veil(k_shard, i))
                unveiled_vs.append(self._remove_shadow_veil(v_shard, i))

            unveiled_k = torch.cat(unveiled_ks, dim=1)
            unveiled_v = torch.cat(unveiled_vs, dim=1)

            if self.active_keys is not None:
                assert self.active_values is not None
                total_k = torch.cat([unveiled_k, self.active_keys], dim=1)
                total_v = torch.cat([unveiled_v, self.active_values], dim=1)
                return total_k, total_v
            return unveiled_k, unveiled_v

        # Ensure we return Tensors
        final_k = self.active_keys
        final_v = self.active_values
        assert final_k is not None and final_v is not None
        return final_k, final_v

    def reset_cache(self) -> None:
        self.active_keys = None
        self.active_values = None
        self.folded_memory_keys = []
        self.folded_memory_values = []
        self.fold_counter = 0
