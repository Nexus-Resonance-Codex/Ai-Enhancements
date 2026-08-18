# MST Lyapunov Clipping (`MSTLyapunovClipping`)

## 1. Overview
Prevents gradient explosion by bounding gradient norms to the maximum Lyapunov exponent:
$$\tilde{g} = g \cdot \min\left(1, \frac{\lambda_{\text{max}}}{\|g\|_2 + \phi^{-2}}\right)$$

## 2. PyTorch Implementation
```python
import torch.nn as nn
from nrc_ai import MSTLyapunovClipping

model = nn.Linear(10, 2)
clipper = MSTLyapunovClipping(max_lyapunov=1.0)
clipper.clip_gradients(model)
```
