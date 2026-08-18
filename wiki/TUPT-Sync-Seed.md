# TUPT Sync Seed (`TUPTSyncSeed`)

## 1. Overview
Synchronizes pseudorandom generation across multi-GPU and multi-node clusters using modular parity checking.

## 2. PyTorch Implementation
```python
from nrc_ai import TUPTSyncSeed

seed = TUPTSyncSeed.generate_sync_seed(epoch=1, rank=0)
```
