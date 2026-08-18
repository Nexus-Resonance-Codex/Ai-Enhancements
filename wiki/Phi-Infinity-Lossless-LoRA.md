# Phi-Infinity Lossless LoRA (`PhiInfinityLosslessLoRA`)

## 1. Overview
Structures Low-Rank Adaptation (LoRA) matrices along self-similar fractal dimensions:
$$\Delta \mathbf{W} = (\mathbf{A} \otimes \mathbf{B}) \cdot \phi^{-r/2}$$

## 2. PyTorch Implementation
```python
import torch
from nrc_ai import PhiInfinityLosslessLoRA

lora = PhiInfinityLosslessLoRA(in_features=512, out_features=512, rank=8)
y = lora(torch.randn(4, 512))
assert y.shape == (4, 512)
```
