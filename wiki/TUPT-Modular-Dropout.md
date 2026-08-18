# TUPT Modular Dropout (`TUPTModularDropout`)

## 1. Overview
Drops activation elements only when their modular index coordinates fall into chaotic residue zones.

## 2. PyTorch Implementation
```python
import torch
from nrc_ai import TUPTModularDropout

dropout = TUPTModularDropout(p=0.1)
y = dropout(torch.randn(2, 64, 256))
```
