# Hodge-Phi Torsion Attention (`HodgePhiTTorsionAttention`)

## 1. Overview & Theoretical Motivation
Standard Multi-Head Attention computes similarity purely via the dot-product $\text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$. In deep networks, this leads to attention entropy collapse where attention distributions degenerate into uniform noise. 

Hodge-Phi Torsion Attention introduces a deterministic golden-ratio torsion field $\mathcal{T}_{\text{Hodge}}(\phi)$ derived from the Hodge dual of the attention tensor:
$$\mathcal{A}_{\phi} = \text{Softmax}\left(\frac{QK^T + \mathcal{T}_{\text{Hodge}}(\phi)}{\sqrt{d_k}}\right) V$$
where $\mathcal{T}_{\text{Hodge}}(\phi) = \phi \cdot \sin(\theta_{\text{QRT}} \cdot (i - j))$ with $\theta_{\text{QRT}} = \arctan(\sqrt{\phi})$.

## 2. Tensor Mechanics
- **Input Tensor:** $X \in \mathbb{R}^{B \times S \times D}$
- **Output Tensor:** $Y \in \mathbb{R}^{B \times S \times D}$, Weights $W \in \mathbb{R}^{B \times H \times S \times S}$
- **Complexity:** $O(S^2 \cdot D)$ with zero dynamic allocation overhead.

## 3. PyTorch Implementation
```python
import torch
from nrc_ai import HodgePhiTTorsionAttention

attn = HodgePhiTTorsionAttention(embed_dim=512, num_heads=8)
x = torch.randn(2, 64, 512)
out, weights = attn(x)
assert out.shape == (2, 64, 512)
```
