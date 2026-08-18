# GTT Entropy Collapse Regularizer (`GTTEntropyCollapseRegularizer`)

## 1. Overview
Dampens representation collapse across deep hidden layers by penalizing latent covariance entropy loss.

## 2. PyTorch Implementation
```python
import torch
from nrc_ai import GTTEntropyCollapseRegularizer

reg = GTTEntropyCollapseRegularizer(weight=0.01)
x = torch.randn(2, 64, 256)
loss = reg(x)
```
