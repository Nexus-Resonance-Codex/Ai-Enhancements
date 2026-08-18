# Resonance Shard KV-Cache (`ResonanceShardKVCache`)

## 1. Overview & Theoretical Motivation
Standard LLMs suffer from $O(N)$ linear memory growth per token in the KV-cache, forcing eviction or lossy quantization. Resonance Shard KV-Cache hierarchically compresses past keys and values into spectral shards scaled by powers of $\phi^{-n}$:
$$\mathbf{K}_{\text{shard}}^{(n)} = \mathbf{K} \cdot \phi^{-n}, \quad \mathbf{V}_{\text{shard}}^{(n)} = \mathbf{V} \cdot \phi^{-n}$$

## 2. PyTorch Implementation
```python
import torch
from nrc_ai import ResonanceShardKVCache

kv = ResonanceShardKVCache(dim=64, num_heads=8, max_shards=16)
k = torch.randn(1, 8, 32, 64)
v = torch.randn(1, 8, 32, 64)
kv.update(k, v)
k_ctx, v_ctx = kv.get_context()
```
