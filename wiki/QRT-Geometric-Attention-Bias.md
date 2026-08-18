# QRT Geometric Attention Bias (`QRTGeometricAttentionBias`)

## 1. Overview
Applies a deterministic quantum resonance wave bias to raw attention logits:
$$\text{Bias}(i, j) = -\frac{|i - j|^2}{\phi} \cdot \cos\left(\frac{\pi}{\phi} |i - j|\right)$$

## 2. PyTorch Implementation
```python
import torch
from nrc_ai import QRTGeometricAttentionBias

bias = QRTGeometricAttentionBias(num_heads=8)
bias_mat = bias(seq_len=64)
assert bias_mat.shape == (1, 8, 64, 64)
```
