# Prime Density Generation (`PrimeDensityConditionedGeneration`)

## 1. Overview
Conditions token sampling probabilities on prime number distribution density curves.

## 2. PyTorch Implementation
```python
import torch
from nrc_ai import PrimeDensityConditionedGeneration

gen = PrimeDensityConditionedGeneration(vocab_size=32000)
logits = torch.randn(1, 32000)
sampled = gen(logits)
```
