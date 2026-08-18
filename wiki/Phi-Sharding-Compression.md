# Phi Sharding Compression (`PhiShardingCompression`)

## 1. Overview
Compresses wide weight matrices along modular golden-ratio coordinate shards:
$$y = \text{LayerNorm}\left(\sum_{i=1}^m \frac{\mathbf{W}_i x}{\phi^i}\right)$$

## 2. PyTorch Implementation
```python
import torch
from nrc_ai import PhiShardingCompression

comp = PhiShardingCompression(in_features=512, out_features=128)
out = comp(torch.randn(4, 512))
assert out.shape == (4, 128)
```
