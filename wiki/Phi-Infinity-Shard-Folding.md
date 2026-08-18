# Phi-Infinity Shard Folding (`PhiInfinityShardFolding`)

## 1. Overview
Recursively folds multi-dimensional activation partitions into a unified residual channel:
$$s_k = x \cdot \phi^k + \text{roll}(x, k) \cdot \phi^{-k}$$

## 2. PyTorch Implementation
```python
import torch
from nrc_ai import PhiInfinityShardFolding

folder = PhiInfinityShardFolding(dim=256, depth=8)
x = torch.randn(2, 64, 256)
folded = folder(x)
assert folded.shape == (2, 64, 256)
```
