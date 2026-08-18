# Phi-Powered Resonant Weighting (`PhiPoweredResonantWeighting`)

## 1. Overview
Weights neural representation channels hierarchically by powers of $\phi^{-i}$.

## 2. PyTorch Implementation
```python
import torch
from nrc_ai import PhiPoweredResonantWeighting

weighting = PhiPoweredResonantWeighting(dim=256)
y = weighting(torch.randn(2, 64, 256))
assert y.shape == (2, 64, 256)
```
