# Golden Flow Normalization (`GoldenFlowNorm`)

## 1. Overview
An optimal alternative to LayerNorm and RMSNorm that normalizes along golden flow vector fields:
$$y = \frac{x}{\|x\|_2 + \phi^{-2}} \odot \gamma + \beta$$

## 2. PyTorch Implementation
```python
import torch
from nrc_ai import GoldenFlowNorm

norm = GoldenFlowNorm(normalized_shape=512)
y = norm(torch.randn(2, 64, 512))
assert y.shape == (2, 64, 512)
```
