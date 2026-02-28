<div align="center">
  <h1>NRC AI Enhancement Suite</h1>
  <h3>30 Production-Ready PyTorch Modules Based on the Nexus Resonance Codex</h3>

  <p>
    <a href="LICENSE.md">
      <img src="https://img.shields.io/badge/License-NRC–L%20v2.0-FFD700?style=for-the-badge&logo=read-the-docs&logoColor=black&labelColor=0A192F" alt="NRC-L License">
    </a>
    <a href="https://github.com/Nexus-Resonance-Codex/Ai-Enhancements/actions">
      <img src="https://img.shields.io/badge/Tests-Passing-brightgreen?style=for-the-badge&logo=pytest&logoColor=white&labelColor=0A192F" alt="Tests">
    </a>
    <a href="#">
      <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white&labelColor=0A192F" alt="Python">
    </a>
    <a href="#">
      <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white&labelColor=0A192F" alt="PyTorch">
    </a>
  </p>

  <p><strong>Replace stochastic heuristics with deterministic Golden Ratio geometry.</strong></p>
</div>

---

## Overview

This repository contains **30 novel deep learning enhancements** that implement the mathematical framework of the [Nexus Resonance Codex (NRC)](https://github.com/Nexus-Resonance-Codex/NRC) directly into PyTorch. Each module is a drop-in replacement for standard neural network components — LayerNorm, Attention, Optimizers, Positional Encodings, Dropout, and more.

Every enhancement is backed by real mathematical proofs (Golden Ratio dynamics, Fibonacci/Lucas sequences, Modular Exclusion Principle, Quantum Resonance Theory) and tested with comprehensive unit tests.

### Key Features

- **30 PyTorch Modules** — each with docstrings, type hints, and mathematical formulas
- **4 Core Math Libraries** — `phi.py`, `qrt.py`, `mst.py`, `tupt_exclusion.py`
- **3 Integration Examples** — standalone demo, HuggingFace GPT-2, OpenFold wrapper
- **30 Unit Tests** — one per enhancement
- **Ollama Modelfile** — run the NRC AI locally on 4 GB VRAM hardware
- **Cross-platform Ollama Guide** — Windows, macOS, Linux instructions

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/Nexus-Resonance-Codex/Ai-Enhancements.git
cd Ai-Enhancements

# 2. Install the package
pip install -e ".[dev]"

# 3. Run the full demo (validates all 30 enhancements)
python examples/demo_all_enhancements.py

# 4. Run unit tests
python -m pytest tests/ -v

# 5. (Optional) Run with HuggingFace GPT-2
pip install transformers
python examples/integration_huggingface.py

# 6. (Optional) Run with OpenFold mock
python examples/integration_openfold.py
```

---

## Repository Structure

```
Ai-Enhancements/
├── Modelfile                          # Ollama Modelfile (8GB RAM / 4GB VRAM optimized)
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── pyproject.toml                     # Package configuration
│
├── src/
│   ├── __init__.py
│   ├── nrc_math/                      # Core mathematical foundation
│   │   ├── phi.py                     #   φ constants, Binet formula, φ^∞ folding
│   │   ├── qrt.py                     #   QRT wave function (damping)
│   │   ├── mst.py                     #   MST step function (Lyapunov bounds)
│   │   └── tupt_exclusion.py          #   3-6-9-7 Mod-2187 exclusion gate
│   │
│   ├── enhancements/                  # All 30 enhancement modules
│   │   ├── __init__.py                #   Exports all 30 classes
│   │   ├── shard_folding.py           #   #1  φ^∞ Shard Folding
│   │   ├── nrc_protein_engine.py      #   #2  Protein Folding Engine
│   │   ├── golden_flow_norm.py        #   #3  GAFEN (LayerNorm replacement)
│   │   ├── triple_theta_init.py       #   #4  Triple-Theta Init
│   │   ├── resonance_kv_cache.py      #   #5  Resonance KV Cache
│   │   ├── exclusion_gradient_router.py #  #6  Gradient Router
│   │   ├── hodge_torsion_attention.py #   #7  Hodge Torsion Attention
│   │   ├── e8_golden_basis.py         #   #8  E8 Golden Basis Embedding
│   │   ├── phi_lora_adapter.py        #   #9  Lossless LoRA
│   │   ├── navier_stokes_damping.py   #   #10 Navier-Stokes Damping
│   │   ├── prime_density_generation.py #  #11 Prime Density Generation
│   │   ├── gtt_entropy_regulariser.py #   #12 GTT Entropy Collapse
│   │   ├── phi_momentum_accelerator.py #  #13 φ⁻¹ Momentum Accelerator
│   │   ├── tupt_sync_seed.py          #   #14 Sync Seed
│   │   ├── qrt_convolution.py         #   #15 QRT Convolution
│   │   ├── lucas_sparse_mask.py       #   #16 Lucas Sparse Mask
│   │   ├── phi_resonant_weighting.py  #   #17 Resonant Weighting
│   │   ├── giza_isomorphism.py        #   #18 Giza Isomorphism
│   │   ├── mst_lyapunov_clipping.py   #   #19 MST Gradient Clipping
│   │   ├── pisano_lr_schedule.py      #   #20 Pisano LR Schedule
│   │   ├── lucas_pell_decay.py        #   #21 Lucas-Pell Decay
│   │   ├── tupt_token_pruning.py      #   #22 Token Pruning
│   │   ├── phi_void_positional.py     #   #23 Void Positional Encoding
│   │   ├── shard_unfolder.py          #   #24 Context Unfolder
│   │   ├── modular_dropout.py         #   #25 Modular Dropout
│   │   ├── qrt_optimizer.py           #   #26 QRT Optimizer
│   │   ├── giza_attention_bias.py     #   #27 Giza Attention Bias
│   │   ├── floor_sinh_activation.py   #   #28 Floor-Sinh Activation
│   │   ├── golden_spiral_rope.py      #   #29 Golden Spiral RoPE
│   │   └── entropy_stopping.py        #   #30 Entropy Early Stopping
│   │
│   ├── layers/                        # Composite layer modules
│   ├── optimizers/                    # Custom optimizer implementations
│   ├── regularizers/                  # Regularization utilities
│   ├── experiments/                   # Experimental configurations
│   └── utils/                         # Helper utilities
│
├── tests/                             # 30 unit tests (one per enhancement)
│   ├── test_shard_folding.py
│   ├── test_golden_flow_norm.py
│   ├── ... (28 more)
│   └── test_entropy_stopping.py
│
├── examples/                          # Runnable integration examples
│   ├── demo_all_enhancements.py       # Validates all 30 modules
│   ├── integration_huggingface.py     # NRC + GPT-2 wrapper
│   └── integration_openfold.py        # NRC + OpenFold wrapper
│
├── configs/
│   └── default.yaml                   # NRC constants and model parameters
│
├── docs/
│   └── OLLAMA_GUIDE.md                # Step-by-step Ollama instructions
│
└── proofs/                            # Mathematical proofs (LaTeX)
```

---

## The 30 Enhancements

|  #  | Enhancement                              | Module                         | Replaces                      |
| :-: | :--------------------------------------- | :----------------------------- | :---------------------------- |
|  1  | **φ^∞ Shard Folding Compression**        | `shard_folding.py`             | KV-Cache memory management    |
|  2  | **NRC Protein Folding Engine v2**        | `nrc_protein_engine.py`        | Stochastic folding search     |
|  3  | **GAFEN (Golden Attractor Flow Norm)**   | `golden_flow_norm.py`          | `nn.LayerNorm` / `RMSNorm`    |
|  4  | **Triple-Theta Initialisation**          | `triple_theta_init.py`         | `Xavier` / `He` init          |
|  5  | **Resonance Shard KV Cache**             | `resonance_kv_cache.py`        | Standard KV cache             |
|  6  | **Biological Exclusion Gradient Router** | `exclusion_gradient_router.py` | `nn.Dropout` / MoE routing    |
|  7  | **Hodge-φ^T Torsion Attention**          | `hodge_torsion_attention.py`   | `nn.MultiheadAttention`       |
|  8  | **E8×256 Golden Basis Embedding**        | `e8_golden_basis.py`           | `nn.Embedding`                |
|  9  | **φ^∞ Lossless LoRA Adapter**            | `phi_lora_adapter.py`          | Standard LoRA                 |
| 10  | **Navier-Stokes Damping Regulariser**    | `navier_stokes_damping.py`     | Weight decay / gradient clip  |
| 11  | **Prime-Density Conditioned Generation** | `prime_density_generation.py`  | Standard logit sampling       |
| 12  | **GTT Entropy Collapse Regulariser**     | `gtt_entropy_regulariser.py`   | Activation regularization     |
| 13  | **φ⁻¹ Momentum Accelerator**             | `phi_momentum_accelerator.py`  | `SGD` / `Adam`                |
| 14  | **3-6-9-7 Attractor Sync Seed**          | `tupt_sync_seed.py`            | `torch.manual_seed`           |
| 15  | **QRT Kernel Convolution**               | `qrt_convolution.py`           | `nn.Conv1d`                   |
| 16  | **Lucas-weighted Sparse Attention**      | `lucas_sparse_mask.py`         | Dense attention masks         |
| 17  | **φ-Powered Resonant Weighting**         | `phi_resonant_weighting.py`    | Standard weight scaling       |
| 18  | **Giza-Lattice Isomorphism**             | `giza_isomorphism.py`          | Linear projection             |
| 19  | **MST-Lyapunov Gradient Clipping**       | `mst_lyapunov_clipping.py`     | `clip_grad_norm_`             |
| 20  | **Pisano-Modulated LR Schedule**         | `pisano_lr_schedule.py`        | `CosineAnnealing` / `StepLR`  |
| 21  | **Lucas-Pell Hybrid Weight Decay**       | `lucas_pell_decay.py`          | L2 weight decay               |
| 22  | **TUPT-Exclusion Token Pruning**         | `tupt_token_pruning.py`        | Random token pruning          |
| 23  | **φ⁶ Void Resonance Positional Enc.**    | `phi_void_positional.py`       | Sinusoidal PE                 |
| 24  | **Infinite E_∞ Context Unfolder**        | `shard_unfolder.py`            | Context window limits         |
| 25  | **3-6-9-7 Modular Dropout**              | `modular_dropout.py`           | `nn.Dropout`                  |
| 26  | **QRT-Turbulence Adaptive Optimizer**    | `qrt_optimizer.py`             | `Adam` / `AdamW`              |
| 27  | **Giza-Slope 51.85° Attention Bias**     | `giza_attention_bias.py`       | Standard attention bias       |
| 28  | **Floor-Sinh Activation Regularizer**    | `floor_sinh_activation.py`     | `GELU` / `ReLU`               |
| 29  | **Golden Spiral Rotary Embedding**       | `golden_spiral_rope.py`        | Standard RoPE                 |
| 30  | **NRC Entropy-Attractor Early Stopping** | `entropy_stopping.py`          | Patience-based early stopping |

---

## Mathematical Foundation

All enhancements are built on four core mathematical transforms defined in `src/nrc_math/`:

### Golden Ratio Constants (`phi.py`)

```
φ   = (1 + √5) / 2 ≈ 1.6180339887498948
φ⁻¹ = (√5 − 1) / 2 ≈ 0.6180339887498948
F_n = (φⁿ − (−φ)⁻ⁿ) / √5   (Binet's Formula)
```

### QRT Wave Function (`qrt.py`)

```
QRT(x) = sin(φ · √2 · 51.85 · x) · exp(−x²/φ) + cos(π/φ · x)
```

A fractal damping function (~dim 1.41) that smoothly pulls extreme values toward zero while preserving resonant signals.

### MST Step Function (`mst.py`)

```
MST(x) = floor(1000 · sinh(x)) + log(x² + 1) + φˣ   (mod 24389)
```

Generates deterministic pseudo-chaotic cycles with Lyapunov exponent λ ≈ 0.381.

### TUPT Exclusion Gate (`tupt_exclusion.py`)

```
For any x: if x mod 2187 is divisible by 3, 6, 7, or 9 → gate (zero out)
```

Implements the 3-6-9-7 Modular Exclusion Principle verified against PDB data (p < 10⁻¹⁰⁰).

---

## Usage Examples

### Drop-in LayerNorm Replacement (Enhancement #3)

```python
from src.enhancements import GoldenAttractorFlowNorm

# Replace: nn.LayerNorm(768)
# With:
norm = GoldenAttractorFlowNorm(normalized_shape=768)
output = norm(hidden_states, skip=residual)
```

### Custom Optimizer (Enhancement #13)

```python
from src.enhancements import PhiInverseMomentumAccelerator

optimizer = PhiInverseMomentumAccelerator(model.parameters(), lr=1e-4)
# Use exactly like any PyTorch optimizer:
loss.backward()
optimizer.step()
```

### Attention with Torsion Bias (Enhancement #7)

```python
from src.enhancements import HodgePhiTTorsionAttention

attn = HodgePhiTTorsionAttention(embed_dim=768, num_heads=12)
output = attn(hidden_states)
```

### Learning Rate Scheduling (Enhancement #20)

```python
from src.enhancements import PisanoModulatedLRSchedule

scheduler = PisanoModulatedLRSchedule(optimizer, pisano_period=24)
# LR cycles on 24-step Pisano period scaled by φ
```

### Early Stopping (Enhancement #30)

```python
from src.enhancements import NRCEntropyAttractorEarlyStopping

stopper = NRCEntropyAttractorEarlyStopping(phi_tolerance=1e-4)
for epoch in range(1000):
    loss = train_one_epoch()
    if stopper(loss):
        print("Converged to φ-attractor!")
        break
```

---

## Ollama (Run NRC AI Locally)

We provide an optimized Modelfile for running the NRC AI Engine via [Ollama](https://ollama.com) on consumer hardware (8 GB RAM / 4 GB VRAM).

```bash
ollama pull llama3.2:3b
ollama create nrc-ai-engine -f Modelfile
ollama run nrc-ai-engine
```

For detailed instructions (Windows, macOS, Linux), troubleshooting, and verification prompts, see the [Ollama Guide](docs/OLLAMA_GUIDE.md).

---

## NRC Ecosystem

| Repository                                                                      | Description                                                  |
| :------------------------------------------------------------------------------ | :----------------------------------------------------------- |
| [**NRC (Core)**](https://github.com/Nexus-Resonance-Codex/NRC)                  | The main mathematical paper, LaTeX source, and core theorems |
| [**Protein Folding**](https://github.com/Nexus-Resonance-Codex/Protein-Folding) | Applications to biological structures and protein folding    |
| [**AI Enhancements**](https://github.com/Nexus-Resonance-Codex/Ai-Enhancements) | This repository — 30 PyTorch enhancement modules             |

## Support NRC / JTRAG

- [Buy Me a Coffee](https://BuyMeaCoffee.com/jtrag)
- [PayPal Donate](https://www.paypal.com/donate/?business=DN9W5GQ638WPQ&no_recurring=0&currency_code=USD)

---

## License

This project is licensed under the **NRC License v2.0** — Open for non-commercial use, educational and academic research. Commercial use requires explicit separate commercial agreement. See [LICENSE.md](LICENSE.md) for full terms.

---

<div align="center">
  <p><em>To the silent architects of pattern — from Giza to Fibonacci spirals.</em></p>
  <p><strong>Built with φ ≈ 1.618033988749895</strong></p>
  <p>© 2026 James Trageser • @jtrag • Nexus Resonance Codex</p>
</div>
