# Documentation: NRC Ai-Enhancements

This repository provides professional-grade AI enhancements derived from the Nexus Resonance Codex. By integrating deterministic $\varphi$-geometry into standard deep learning components, we achieve unprecedented stability and scaling.

## Core Pillars

### 1. $\varphi^\infty$ Spiral Memory ($O(1)$ Context)
The infinite context capability is achieved through **Shard Folding**. Instead of keeping all attention keys in VRAM, we fold them into a persistent golden-spiral lattice.
- **Reference**: `src/nrc_ai/memory.py`

### 2. TTT-Compliant Gradient Routing
We use the **Triple Theta Theorem (TTT)** to route gradients around chaotic modular voids. This prevents the "entropy collapse" often seen in extremely deep transformers.
- **Reference**: `src/nrc_ai/exclusion_gradient_router.py`

### 3. Resonant Weighting & Initialization
By initializing weights to the **E8 Golden Basis**, we ensure that the network starts in a state of maximal geometric coherence.
- **Reference**: `src/nrc_ai/e8_golden_basis.py`

## Integration Guide

### Requirements
- Python 3.12+
- PyTorch 2.2+
- `nrc-math` (Professional Core)

### Installation
```bash
uv pip install -e .
```

### Basic Usage
```python
import torch
from nrc_ai import SpiralMemory

# Initialize 512D Spiral Memory
memory = SpiralMemory(hidden_dim=512)

# Update with new tokens
for token_embedding in batch:
    state = memory.update(token_embedding)
    
print(f"Lattice Resonance: {memory.retrieve().mean():.4f}")
```

---

*Verified by the Nexus Resonance Codex AI Division (2026).*
