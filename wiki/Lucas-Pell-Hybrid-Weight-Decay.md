# Lucas-Pell Hybrid Weight Decay (`LucasPellHybridWeightDecay`)

## 1. Overview
Scales weight decay dynamically based on the layer's harmonic coordinate in the Lucas-Pell sequence:
$$\lambda_w = \lambda_0 \cdot \left(\frac{L_k}{P_k + \phi^{-2}}\right)$$

## 2. PyTorch Implementation
```python
from nrc_ai import LucasPellHybridWeightDecay
import torch.nn as nn

model = nn.Linear(10, 2)
decay = LucasPellHybridWeightDecay(base_decay=1e-4)
loss = decay(model.parameters())
```
