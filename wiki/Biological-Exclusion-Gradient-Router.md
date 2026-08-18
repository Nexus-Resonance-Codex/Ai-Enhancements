# Biological Exclusion Gradient Router (`BiologicalExclusionGradientRouter`)

## 1. Overview
An autograd Function that zeroes out backward gradients whose modular indices fall in the chaotic void $\{0, 3, 6, 9\} \pmod 9$.

## 2. PyTorch Implementation
```python
import torch
from nrc_ai import BiologicalExclusionGradientRouter

router = BiologicalExclusionGradientRouter()
x = torch.randn(4, 16, requires_grad=True)
y = router(x)
y.sum().backward()
```
