# Phi-Inverse Momentum Accelerator (`PhiInverseMomentumAccelerator`)

## 1. Overview
Replaces empirical momentum $\beta=0.9$ with the provable golden ratio attractor $\phi^{-1} \approx 0.61803398875$:
$$v_t = \phi^{-1} v_{t-1} + (1 - \phi^{-1}) g_t$$

## 2. PyTorch Implementation
```python
import torch.nn as nn
from nrc_ai import PhiInverseMomentumAccelerator

model = nn.Linear(10, 2)
opt = PhiInverseMomentumAccelerator(model.parameters(), lr=1e-3)
```
