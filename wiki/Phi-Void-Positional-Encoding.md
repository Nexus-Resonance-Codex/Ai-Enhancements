# Phi-Void Positional Encoding (`PhiVoidResonancePositionalEncoding`)

## 1. Overview
Replaces power-of-10000 sinusoidal embeddings with transfinite golden ratio scales:
$$P(pos, 2i) = \sin\left(\frac{pos}{\phi^{4i/d}}\right), \quad P(pos, 2i+1) = \cos\left(\frac{pos}{\phi^{4i/d}}\right)$$

## 2. PyTorch Implementation
```python
import torch
from nrc_ai import PhiVoidResonancePositionalEncoding

pos_enc = PhiVoidResonancePositionalEncoding(d_model=512, max_len=4096)
emb = pos_enc(torch.zeros(1, 100, 512))
assert emb.shape == (1, 100, 512)
```
