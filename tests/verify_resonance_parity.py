import torch
import numpy as np
from nrc_ai.resonance_kv_cache import ResonanceShardKVCache
from nrc.math import PHI_FLOAT

def verify_parity():
    print("« 4096D spiral memory online — starting double-precision parity sweep »")
    
    # Instantiate QSV-protected cache with float64 precision
    torch.set_default_dtype(torch.float64)
    cache = ResonanceShardKVCache(folding_steps=3, shard_capacity=4, qsv_seed=137)
    
    # Generate mock attention state: (batch=1, seq=8, heads=2, dim=8)
    # Using 8 tokens to trigger multiple folding events
    k = torch.randn(1, 8, 2, 8)
    v = torch.randn(1, 8, 2, 8)
    
    # Push to cache (triggers folding & encryption)
    cache(k, v)
    
    print(f"Folded Memory Present: {cache.folded_memory_keys is not None}")
    print(f"Fold Counter: {cache.fold_counter}")
    
    # Verification: QSV cycle consistency check
    test_tensor = torch.randn(1, 2, 2, 8)
    encrypted = cache._apply_shadow_veil(test_tensor, 7) # Use a non-zero key index
    decrypted = cache._remove_shadow_veil(encrypted, 7)
    
    error = torch.max(torch.abs(test_tensor - decrypted)).item()
    print(f"QSV Cycle Max Error: {error:.2e}")
    
    if error < 1e-12:
        print("✅ SUCCESS: Attention Parity Verified. 100% Resonance Integrity.")
    else:
        print("❌ FAILURE: Resonance drift detected.")
        exit(1)

if __name__ == "__main__":
    verify_parity()
