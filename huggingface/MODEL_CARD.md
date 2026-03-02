---
language:
  - en
license: other
license_name: nrc-license-2.0
license_link: https://github.com/Nexus-Resonance-Codex/Ai-Enhancements/blob/main/LICENSE.md
tags:
  - pytorch
  - deep-learning
  - golden-ratio
  - mathematics
  - resonance
  - attention
  - transformers
  - optimization
  - research
  - nrc
pipeline_tag: text-generation
library_name: nrc_ai
model_type: nrc-enhancement-modules
---

<div align="center">

# NRC AI Enhancements

**30 Mathematically Rigid PyTorch Enhancement Modules**
_Based on the Nexus Resonance Codex — replacing stochastic heuristics with Golden Ratio geometry_

[![GitHub](https://img.shields.io/badge/GitHub-ai--enhancements-181717?logo=github)](https://github.com/Nexus-Resonance-Codex/Ai-Enhancements)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-NRC--L--2.0-gold)](LICENSE.md)

</div>

---

## Overview

The NRC AI Enhancements library provides **30 drop-in PyTorch modules** that replace conventional stochastic deep learning operations with **deterministic, geometrically grounded** alternatives derived from the Nexus Resonance Codex mathematical framework.

The foundational principle: **the Golden Ratio φ = 1.6180339887...** is not merely aesthetic — it is a structural attractor that governs stable energy distributions in physics, biology, and information systems. These enhancements embed that structure directly into neural network operations.

```python
pip install "nrc_ai @ git+https://github.com/Nexus-Resonance-Codex/Ai-Enhancements.git"
```

---

## Mathematical Core

All 30 modules derive from four foundational equations:

### 1. Quantum Resonance Theory (QRT)

Replaces stochastic dropout with fractal wave damping:

```
QRT(x) = sin(φ · √2 · 51.85° · x) · exp(−x²/φ) + cos(π/φ · x)
```

### 2. Modular Synchronisation Theory (MST)

Replaces Adam gradient noise with Lyapunov-bounded cyclic steps:

```
x_{n+1} = |floor(1000·sinh(x_n)) + log(x_n²+1) + φ^{x_n}| mod 24389
```

_Cycle length: ~2100 phases, Lyapunov exponent λ ≈ 0.381_

### 3. TUPT Exclusion Gate (3-6-9-7)

Biologically-informed signal gating:

```
TUPT(x) = 0  if  (x mod 2187) divisible by {3, 6, 7, 9}
         x  otherwise
```

### 4. 2048D Phi-Lattice Projection

Maps any signal into the Golden Ratio hyperdimensional space:

```
L_i(x) = x · φ^{-i/2048} · cos(i · Giza_rad)
```

---

## All 30 Enhancements

| #   | Module Class                         | Replaces          | Key Formula                |
| :-- | :----------------------------------- | :---------------- | :------------------------- |
| 1   | `PhiInfinityShardFolding`            | Attention         | φ^∞ shard topology         |
| 2   | `NRCProteinFoldingEngine`            | Scaffold          | 2048D lattice TUPT filter  |
| 3   | `GoldenAttractorFlowNorm`            | LayerNorm         | φ-attractor normalization  |
| 4   | `TripleThetaInitializer`             | Xavier Init       | 3θ resonance seed          |
| 5   | `ResonanceShardKVCache`              | KV-Cache          | φ^n memory sharding        |
| 6   | `BiologicalExclusionGradientRouter`  | Gradient clipping | TUPT mod-9 gate            |
| 7   | `HodgePhiTTorsionAttention`          | Self-Attention    | Hodge torsion biasing      |
| 8   | `E8GoldenBasisEmbedding`             | Embedding         | E8 root basis + φ          |
| 9   | `PhiInfinityLosslessLoRA`            | LoRA              | φ^∞ lossless adapter       |
| 10  | `NavierStokesDampingRegularizer`     | L2 Regularizer    | NS fractional damping      |
| 11  | `PrimeDensityConditionedGeneration`  | Sampling          | Prime density seeds        |
| 12  | `GTTEntropyCollapseRegularizer`      | Entropy penalty   | GTT threshold collapse     |
| 13  | `PhiInverseMomentumAccelerator`      | Momentum          | φ⁻¹ velocity scaling       |
| 14  | `TUPTAttractorSyncSeed`              | RNG Seed          | TUPT cycle sync            |
| 15  | `QRTKernelConvolution`               | Conv1D/2D         | QRT wave kernel            |
| 16  | `LucasWeightedSparseAttention`       | Sparse Attn       | Lucas number masking       |
| 17  | `PhiPoweredResonantWeighting`        | Weight init       | φ^n spectral decay         |
| 18  | `GizaLatticeIsomorphism`             | Projection        | 51.85° slope map           |
| 19  | `MSTLyapunovGradientClipping`        | Grad clip         | MST λ≈0.381 bound          |
| 20  | `PisanoModulatedLRSchedule`          | LR Schedule       | Pisano period cycle        |
| 21  | `LucasPellHybridWeightDecay`         | Weight Decay      | Lucas-Pell recursion       |
| 22  | `TUPTExclusionTokenPruning`          | Token pruning     | Mod-9 pruning gate         |
| 23  | `PhiVoidResonancePositionalEncoding` | RoPE/APE          | φ-void sinusoidal PE       |
| 24  | `InfiniteEInfinityContextUnfolder`   | Context window    | E∞ recursive unfolding     |
| 25  | `TUPTModularDropout`                 | Dropout           | TUPT-gated structural drop |
| 26  | `QRTTurbulenceOptimizer`             | Adam/AdaGrad      | QRT turbulence gradient    |
| 27  | `GizaSlopeAttentionBias`             | Attention bias    | 51.85° Giza weighting      |
| 28  | `FloorSinhActivation`                | ReLU/GELU         | floor(sinh(x)) + φ term    |
| 29  | `GoldenSpiralRotaryEmbedding`        | RoPE              | φ-spiral rotation matrix   |
| 30  | `NRCEntropyAttractorEarlyStopping`   | Early stopping    | NRC entropy convergence    |

---

## Quick Start

```python
import torch
from nrc_ai import GoldenAttractorFlowNorm, QRTTurbulenceOptimizer

model = torch.nn.Linear(512, 512)

# Drop in Golden-Ratio normalized layer
norm = GoldenAttractorFlowNorm(normalized_shape=512)

# Replace Adam with QRT turbulence optimizer
optimizer = QRTTurbulenceOptimizer(model.parameters(), lr=1e-3)
```

---

## Installation

```bash
# Install with the nrc core library
pip install "nrc @ git+https://github.com/Nexus-Resonance-Codex/NRC.git"
pip install "nrc_ai @ git+https://github.com/Nexus-Resonance-Codex/Ai-Enhancements.git"

# Or clone and install locally (recommended for development)
git clone https://github.com/Nexus-Resonance-Codex/Ai-Enhancements.git
cd Ai-Enhancements
./setup_venv.sh
source .venv/bin/activate
```

---

## Citation

```bibtex
@software{nrc_ai_2026,
  author  = {Trageser, James},
  title   = {NRC AI Enhancements: Golden Ratio PyTorch Modules},
  year    = {2026},
  url     = {https://github.com/Nexus-Resonance-Codex/Ai-Enhancements},
  version = {1.0.0}
}
```
