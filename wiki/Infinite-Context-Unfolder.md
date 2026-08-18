# Infinite Context Unfolder (`InfiniteEInfinityContextUnfolder`)

## 1. Overview
The mathematical inversion of shard folding, reconstructing compressed historical latent states back to explicit sequence representations:
$$\hat{x} = \sum_{k=1}^N s_k \cdot \phi^{-2k}$$

## 2. PyTorch Implementation
```python
import torch
from nrc_ai import InfiniteEInfinityContextUnfolder

unfolder = InfiniteEInfinityContextUnfolder(dim=256, depth=8)
restored = unfolder(folded)
assert restored.shape == (2, 64, 256)
```
