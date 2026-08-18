# NRC Protein Folding Engine (`NRCProteinFoldingEngine`)

## 1. Overview
Interfaces structural sequence embeddings with physical geometry energy minimization under deterministic $\phi$-potentials.

## 2. PyTorch Implementation
```python
import torch
from nrc_ai import NRCProteinFoldingEngine

bio_engine = NRCProteinFoldingEngine(d_model=256)
x = torch.randn(1, 100, 256)
coords = bio_engine(x)
```
