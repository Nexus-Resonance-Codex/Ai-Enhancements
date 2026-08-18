# NRC Entropy Attractor Early Stopping (`NRCEntropyAttractorEarlyStopping`)

## 1. Overview
Monitors validation loss convergence against Lyapunov stabilization thresholds, stopping training when theoretical resonance is achieved.

## 2. PyTorch Implementation
```python
from nrc_ai import NRCEntropyAttractorEarlyStopping

stopper = NRCEntropyAttractorEarlyStopping(patience=5, min_delta=1e-4)
should_stop = stopper(val_loss=0.024)
```
