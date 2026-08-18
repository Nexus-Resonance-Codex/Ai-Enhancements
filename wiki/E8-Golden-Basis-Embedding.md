# E8 Golden Basis Embedding (`E8GoldenBasisEmbedding`)

## 1. Overview
Maps input tokens onto the discrete roots of the $E_8$ exceptional Lie group scaled by powers of $\phi$.

## 2. PyTorch Implementation
```python
import torch
from nrc_ai import E8GoldenBasisEmbedding

emb = E8GoldenBasisEmbedding(num_embeddings=1000, embedding_dim=256)
tokens = torch.randint(0, 1000, (2, 16))
vecs = emb(tokens)
assert vecs.shape == (2, 16, 256)
```
