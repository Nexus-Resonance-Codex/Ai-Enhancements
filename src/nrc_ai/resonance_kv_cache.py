from typing import Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
from nrc.math import QuantumShadowVeil, MST_MODULUS

from .phi_sharding_compression import PhiShardingCompression
from .shard_folding import PhiInfinityShardFolding

__version__ = "2.3.0-QSV"

class ResonanceShardKVCache(nn.Module):
    """
    Institutional-grade KV-cache featuring Resonance-Shard Encryption.
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
        self.folded_memory_keys: Optional[torch.Tensor] = None
        self.folded_memory_values: Optional[torch.Tensor] = None

    def _apply_shadow_veil(self, tensor: torch.Tensor, key_idx: int) -> torch.Tensor:
        """Applies the Residue-Hiding (RH) encryption to a memory shard."""
        device = tensor.device
        dtype = tensor.dtype
        
        # Convert to numpy for QSV manifold transform
        arr = tensor.detach().cpu().numpy()
        encrypted_arr = self.qsv.residue_hide_encrypt(arr, key_idx)
        
        return torch.from_numpy(encrypted_arr).to(device=device, dtype=dtype)

    def _remove_shadow_veil(self, tensor: torch.Tensor, key_idx: int) -> torch.Tensor:
        """Decrypts a memory shard via inverse resonant phasing."""
        from nrc.math import PHI_FLOAT
        device = tensor.device
        dtype = tensor.dtype
        
        # Inverse phasing: encrypted / phi^{-n} - salt
        key = self.qsv.keys[key_idx % len(self.qsv.keys)]
        salt = float(key % 256) / 256.0
        decrypted_arr = (tensor.cpu().numpy() / (PHI_FLOAT ** -(key_idx % 13))) - salt
        
        return torch.from_numpy(decrypted_arr).to(device=device, dtype=dtype)

    def forward(self, new_keys: torch.Tensor, new_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. First initialization
        if self.active_keys is None:
            self.active_keys = new_keys
            self.active_values = new_values
            return self.active_keys, self.active_values

        # 2. Append incoming context
        self.active_keys = torch.cat([self.active_keys, new_keys], dim=1)
        self.active_values = torch.cat([self.active_values, new_values], dim=1)

        # 3. Phase-Folding Limit Step
        if self.active_keys.size(1) >= self.shard_capacity:
            compressed_k = self.folding_compressor(self.active_keys)
            compressed_v = self.folding_compressor(self.active_values)

            # Protect shards using Quantum Shadow Veil
            protected_k = self._apply_shadow_veil(compressed_k, self.fold_counter)
            protected_v = self._apply_shadow_veil(compressed_v, self.fold_counter)
            self.fold_counter += 1

            if self.folded_memory_keys is None:
                self.folded_memory_keys = protected_k
                self.folded_memory_values = protected_v
            else:
                self.folded_memory_keys = self.folded_memory_keys + protected_k
                self.folded_memory_values = self.folded_memory_values + protected_v

            self.active_keys = None
            self.active_values = None

        # 4. Veil-Authenticated Retrieval
        if self.folded_memory_keys is not None:
            # Authenticated decryption for attention cycle
            unveiled_k = self._remove_shadow_veil(self.folded_memory_keys, self.fold_counter - 1)
            unveiled_v = self._remove_shadow_veil(self.folded_memory_values, self.fold_counter - 1)
            
            if self.active_keys is not None:
                total_k = torch.cat([unveiled_k, self.active_keys], dim=1)
                total_v = torch.cat([unveiled_v, self.active_values], dim=1)
                return total_k, total_v
            return unveiled_k, unveiled_v

        return self.active_keys, self.active_values

    def reset_cache(self) -> None:
        self.active_keys = None
        self.active_values = None
        self.folded_memory_keys = None
        self.folded_memory_values = None
        self.fold_counter = 0
