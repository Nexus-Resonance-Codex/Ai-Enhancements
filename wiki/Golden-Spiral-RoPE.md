# Golden Spiral Rotary Embeddings (`GoldenSpiralRotaryEmbedding`)

## 1. Overview & Theoretical Motivation
Standard Rotary Position Embedding (RoPE) uses base frequencies $10000^{-2i/d}$, leading to high-frequency aliasing during context extension. Golden Spiral RoPE scales rotation frequencies along the golden angle $\theta = \frac{360^\circ}{\phi^2} \approx 137.507764^\circ$.

## 2. Mathematical Formulation
$$\mathbf{R}_{\theta}(m) = \text{diag}\left(R(\theta_1 m), R(\theta_2 m), \dots, R(\theta_{d/2} m)\right), \quad \theta_k = \frac{360^\circ}{\phi^2} \cdot \phi^{-2k/d}$$

## 3. PyTorch Implementation
```python
import torch
from nrc_ai import GoldenSpiralRotaryEmbedding

rope = GoldenSpiralRotaryEmbedding(dim=64, max_seq_len=8192)
q = torch.randn(2, 8, 128, 64)
q_rot = rope(q)
assert q_rot.shape == (2, 8, 128, 64)
```
