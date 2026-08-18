# Nexus Resonance Codex: AI Enhancements (nrc-ai)

<p align="center">
  <img src="https://raw.githubusercontent.com/Nexus-Resonance-Codex/.github/main/profile/nrc_logo.png" alt="NRC AI Enhancements Logo" width="380">
</p>

<p align="center">
  <strong>Deterministic Deep Learning & Cognitive Resonance Architecture for High-Dimensional Foundation Models</strong>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-AGPL--3.0-blue.svg?style=flat-square" alt="AGPL-3.0 License"></a>
  <a href="LICENSE-DATA"><img src="https://img.shields.io/badge/Data%20License-CC%20BY--NC--SA%204.0-lightgrey.svg?style=flat-square" alt="CC BY-NC-SA 4.0"></a>
  <a href="https://github.com/Nexus-Resonance-Codex/Ai-Enhancements/wiki"><img src="https://img.shields.io/badge/Wiki-Institutional%20Docs-blueviolet.svg?style=flat-square" alt="GitHub Wiki"></a>
  <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB.svg?style=flat-square&logo=python&logoColor=white" alt="Python Versions">
  <img src="https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C.svg?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch Supported">
  <img src="https://img.shields.io/badge/Stability-TTT--7%20Verified-008080.svg?style=flat-square" alt="TTT-7 Verified">
</p>

---

## Executive Overview

The **`Ai-Enhancements` (`nrc-ai`)** repository contains the production neural architecture suite of the Nexus Resonance Codex (NRC). It replaces standard stochastic heuristics (such as unconstrained Gaussian initialization, random dropout, quadratic KV caching, and heuristic Adam momentum) with mathematically rigid **Golden Ratio ($\phi$) geometry**, **modular residue exclusion (TUPT)**, and **Lyapunov-bounded stability manifolds (MST / TTT-7)**.

Across 30+ PyTorch modules, `nrc-ai` provides drop-in replacements for standard attention mechanisms, rotary embeddings, optimizers, learning rate schedulers, normalization layers, and KV-cache managers. These enhancements eliminate representation collapse, prevent loss explosion during long-context training, and enable functionally unbounded memory retention.

---

## Architectural Taxonomy

```
+---------------------------------------------------------------------------------------------------+
|                                     NRC-AI NEURAL TAXONOMY                                        |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  1. Attention & Positional Encoding           2. Context Memory, KV-Cache & Compression           |
|  +-------------------------------------+      +--------------------------------------------+      |
|  | - HodgePhiTTorsionAttention         |      | - ResonanceShardKVCache                    |      |
|  | - LucasWeightedSparseAttention      | <--> | - PhiInfinityShardFolding                  |      |
|  | - GoldenSpiralRotaryEmbedding       |      | - InfiniteEInfinityContextUnfolder         |      |
|  | - PhiVoidResonancePositionalEncoding|      | - PhiShardingCompression                   |      |
|  | - QRTGeometricAttentionBias         |      | - PhiInfinityPersistentMemory / ExecAgent  |      |
|  +-------------------------------------+      +--------------------------------------------+      |
|                    ^                                                 ^                            |
|                    |                                                 |                            |
|                    v                                                 v                            |
|  3. Optimizers, Schedulers & Gradients        4. Layers, Encodings & Activations                  |
|  +-------------------------------------+      +--------------------------------------------+      |
|  | - QRTTurbulenceOptimizer            |      | - E8GoldenBasisEmbedding                   |      |
|  | - PhiInverseMomentumAccelerator     |      | - FloorSinhActivation                      |      |
|  | - PisanoModulatedLRSchedule         | <--> | - GoldenFlowNorm                           |      |
|  | - LucasPellHybridWeightDecay        |      | - PhiInfinityLosslessLoRA                  |      |
|  | - MSTLyapunovClipping               |      | - PhiPoweredResonantWeighting              |      |
|  | - BiologicalExclusionGradientRouter |      | - TUPTModularDropout / TokenPruning        |      |
|  | - GTTEntropyCollapseRegularizer     |      | - TripleThetaInitializer                   |      |
|  | - NavierStokesDampingRegularizer    |      | - PrimeDensityConditionedGeneration        |      |
|  | - NRCEntropyAttractorEarlyStopping  |      | - TUPTSyncSeed / GeometricIsomorphism      |      |
|  +-------------------------------------+      +--------------------------------------------+      |
|                                                                                                   |
+---------------------------------------------------------------------------------------------------+
```

---

## 1. Attention & Positional Operators

### 1.1 Hodge-Phi Torsion Attention (`HodgePhiTTorsionAttention`)
- **Mathematical Anchor:** 
  $$\mathcal{A}_{\phi} = \text{Softmax}\left(\frac{Q K^T + \mathcal{T}_{\text{Hodge}}(\phi)}{\sqrt{d_k}}\right) V$$
  where $\mathcal{T}_{\text{Hodge}}(\phi) = \phi^{-13} \cdot \arctan(\sqrt{\phi}) \cdot \mathbf{J}$ introduces an orthogonal geometric phase twist.
- **Intuitive Explanation:** Standard dot-product attention computes token similarity along flat linear projections. Hodge-Phi Torsion Attention injects a deterministic golden-ratio torsion field that models multi-scale harmonic relationships across heads, preventing attention entropy collapse in deep models.
- **Usage Example:**
  ```python
  import torch
  from nrc_ai import HodgePhiTTorsionAttention

  attn = HodgePhiTTorsionAttention(embed_dim=512, num_heads=8)
  x = torch.randn(2, 64, 512)
  out, weights = attn(x)
  print("Attention output shape:", out.shape)  # torch.Size([2, 64, 512])
  ```

### 1.2 Lucas Weighted Sparse Attention (`LucasWeightedSparseAttention`)
- **Mathematical Anchor:** 
  $$\mathcal{M}_{i, j} = \begin{cases} 1 & \text{if } (i - j) \pmod 9 \in \{1, 2, 4, 5, 7, 8\} \text{ and } |i - j| \in \mathcal{L} \\ 0 & \text{otherwise} \end{cases}$$
  where $\mathcal{L} = \{1, 3, 4, 7, 11, 18, 29, 47, \dots\}$ represents the Lucas number sequence.
- **Intuitive Explanation:** Rather than using arbitrary sliding windows or random block sparsity, this operator connects tokens along non-colliding Lucas harmonic frequencies, reducing quadratic $O(N^2)$ memory to $O(N \log N)$ while maintaining full global receptive fields.
- **Usage Example:**
  ```python
  from nrc_ai import LucasWeightedSparseAttention

  sparse_attn = LucasWeightedSparseAttention(embed_dim=256, num_heads=4)
  x = torch.randn(1, 128, 256)
  out = sparse_attn(x)
  ```

### 1.3 Golden Spiral Rotary Position Embedding (`GoldenSpiralRotaryEmbedding`)
- **Mathematical Anchor:** 
  $$\mathbf{R}_{\theta}(m) = \text{diag}\left(R(\theta_1 m), R(\theta_2 m), \dots, R(\theta_{d/2} m)\right), \quad \theta_k = \frac{360^\circ}{\phi^2} \cdot \phi^{-2k/d}$$
- **Intuitive Explanation:** Upgrades standard RoPE by scaling rotation angles along the golden spiral angle ($\theta \approx 137.507764^\circ$). It ensures that relative token distances remain self-similar across arbitrary sequence lengths without frequency aliasing.
- **Usage Example:**
  ```python
  from nrc_ai import GoldenSpiralRotaryEmbedding

  rope = GoldenSpiralRotaryEmbedding(dim=64, max_seq_len=8192)
  q = torch.randn(2, 8, 128, 64)
  q_rot = rope(q)
  ```

### 1.4 Phi-Void Positional Encoding (`PhiVoidResonancePositionalEncoding`)
- **Mathematical Anchor:** 
  $$P(pos, 2i) = \sin\left(\frac{pos}{\phi^{4i/d}}\right), \quad P(pos, 2i+1) = \cos\left(\frac{pos}{\phi^{4i/d}}\right)$$
- **Intuitive Explanation:** Replaces standard power-of-10000 sinusoidal embeddings with transfinite golden ratio scales, preventing high-frequency representation decay in state-space representations.
- **Usage Example:**
  ```python
  from nrc_ai import PhiVoidResonancePositionalEncoding

  pos_enc = PhiVoidResonancePositionalEncoding(d_model=512, max_len=4096)
  emb = pos_enc(torch.zeros(1, 100, 512))
  ```

### 1.5 QRT Geometric Attention Bias (`QRTGeometricAttentionBias`)
- **Mathematical Anchor:** 
  $$\text{Bias}(i, j) = -\frac{|i - j|^2}{\phi} \cdot \cos\left(\frac{\pi}{\phi} |i - j|\right)$$
- **Intuitive Explanation:** Applies a deterministic quantum-resonance wave penalty to the attention logits, regularizing local token interactions and suppressing long-range noise without requiring artificial attention masking.
- **Usage Example:**
  ```python
  from nrc_ai import QRTGeometricAttentionBias

  bias_layer = QRTGeometricAttentionBias(num_heads=8)
  bias_matrix = bias_layer(seq_len=64)
  ```

---

## 2. Context Memory, KV-Cache & Compression

### 2.1 Resonance Shard KV-Cache (`ResonanceShardKVCache`)
- **Mathematical Anchor:** 
  $$\mathbf{K}_{\text{shard}}^{(n)} = \mathbf{K} \cdot \phi^{-n}, \quad \mathbf{V}_{\text{shard}}^{(n)} = \mathbf{V} \cdot \phi^{-n}, \quad n \in \{1, \dots, \text{depth}\}$$
- **Intuitive Explanation:** Rather than evicting tokens when KV-cache limits are reached, older key-value tensors are compressed into hierarchical spectral shards. Retrieval from older history occurs in $O(1)$ time by projecting queries onto the corresponding $\phi^{-n}$ frequency tier.
- **Usage Example:**
  ```python
  from nrc_ai import ResonanceShardKVCache

  kv_cache = ResonanceShardKVCache(dim=64, num_heads=8, max_shards=16)
  k = torch.randn(1, 8, 32, 64)
  v = torch.randn(1, 8, 32, 64)
  kv_cache.update(k, v)
  k_all, v_all = kv_cache.get_context()
  ```

### 2.2 Phi-Infinity Shard Folding (`PhiInfinityShardFolding`)
- **Mathematical Anchor:** 
  $$s_k = x \cdot \phi^k + \text{roll}(x, k) \cdot \phi^{-k}$$
- **Intuitive Explanation:** Recursively folds multi-dimensional activation partitions into a unified residual channel, allowing models to store historical conversational context at high information densities without memory explosion.
- **Usage Example:**
  ```python
  from nrc_ai import PhiInfinityShardFolding

  folder = PhiInfinityShardFolding(dim=256, depth=8)
  tensor = torch.randn(2, 64, 256)
  folded = folder(tensor)
  ```

### 2.3 Infinite Context Unfolder (`InfiniteEInfinityContextUnfolder`)
- **Mathematical Anchor:** 
  $$\hat{x} = \sum_{k=1}^{N} s_k \cdot \phi^{-2k}$$
- **Intuitive Explanation:** The exact mathematical inversion of shard folding. It reconstructs compressed historical latent states back into explicit token sequences with minimal reconstruction error.
- **Usage Example:**
  ```python
  from nrc_ai import InfiniteEInfinityContextUnfolder

  unfolder = InfiniteEInfinityContextUnfolder(dim=256, depth=8)
  restored = unfolder(folded)
  ```

### 2.4 Phi Sharding Compression (`PhiShardingCompression`)
- **Mathematical Anchor:** 
  $$y = \text{LayerNorm}\left(\sum_{i=1}^m \frac{\mathbf{W}_i x}{\phi^i}\right)$$
- **Intuitive Explanation:** Compresses wide weight and projection matrices by grouping parameters along modular golden-ratio coordinate shards, shrinking model footprint while retaining full expressive rank.
- **Usage Example:**
  ```python
  from nrc_ai import PhiShardingCompression

  compressor = PhiShardingCompression(in_features=512, out_features=128)
  out = compressor(torch.randn(4, 512))
  ```

### 2.5 Phi-Infinity Persistent Memory & Executive Agent (`PhiInfinityPersistentMemory`, `ExecutiveAgent`)
- **Mathematical Anchor:** High-dimensional associative matrix retention scaled by continuous Lyapunov decay.
- **Intuitive Explanation:** Maintains long-term agent state and episodic task context across arbitrary conversation turns, routing subtasks to optimal solver modules without context loss.
- **Usage Example:**
  ```python
  from nrc_ai import ExecutiveAgent, PhiInfinityPersistentMemory

  memory = PhiInfinityPersistentMemory(memory_dim=512)
  memory.store("session_1", torch.randn(512))
  retrieved = memory.recall("session_1")
  ```

---

## 3. Optimizers, Schedulers & Gradient Regularization

### 3.1 QRT Turbulence Optimizer (`QRTTurbulenceOptimizer`)
- **Mathematical Anchor:** 
  $$\theta_{t+1} = \theta_t - \eta_t \left(\frac{m_t}{\sqrt{v_t} + \epsilon}\right) \cdot \exp\left(-\frac{\|\nabla \mathcal{L}\|^2}{\phi}\right)$$
- **Intuitive Explanation:** A PyTorch optimizer that models gradient dynamics as turbulent kinetic energy fields. When gradients spike or encounter noisy loss plateaus, exponential fractal damping stabilizes the update trajectory, eliminating divergence.
- **Usage Example:**
  ```python
  import torch.nn as nn
  from nrc_ai import QRTTurbulenceOptimizer

  model = nn.Linear(10, 2)
  optimizer = QRTTurbulenceOptimizer(model.parameters(), lr=1e-3, phi_damping=1.618)
  ```

### 3.2 Phi-Inverse Momentum Accelerator (`PhiInverseMomentumAccelerator`)
- **Mathematical Anchor:** 
  $$v_t = \phi^{-1} v_{t-1} + (1 - \phi^{-1}) g_t, \quad \phi^{-1} \approx 0.61803398875$$
- **Intuitive Explanation:** Traditional momentum parameters ($\beta = 0.9$) are empirically chosen heuristics. This optimizer anchors momentum directly to the golden attractor $\phi^{-1}$, provably minimizing oscillations near ill-conditioned ravines.
- **Usage Example:**
  ```python
  from nrc_ai import PhiInverseMomentumAccelerator

  optimizer = PhiInverseMomentumAccelerator(model.parameters(), lr=1e-3)
  ```

### 3.3 Pisano Modulated Learning Rate Schedule (`PisanoModulatedLRSchedule`)
- **Mathematical Anchor:** 
  $$\eta_t = \eta_{\text{min}} + (\eta_{\text{max}} - \eta_{\text{min}}) \cdot \frac{F_{t \pmod{\pi(m)}}}{\max(F)}$$
  where $\pi(m)$ is the Pisano period for modulo $m$.
- **Intuitive Explanation:** Replaces standard cosine or linear warmups with cyclic Pisano sequence modulations. Periodic harmonic resets allow gradient descent to escape local minima and saddle points deterministically.
- **Usage Example:**
  ```python
  from nrc_ai import PisanoModulatedLRSchedule

  scheduler = PisanoModulatedLRSchedule(optimizer, modulo=9, base_lr=1e-3)
  ```

### 3.4 Lucas-Pell Hybrid Weight Decay (`LucasPellHybridWeightDecay`)
- **Mathematical Anchor:** 
  $$\lambda_w = \lambda_0 \cdot \left(\frac{L_k}{P_k + \phi^{-2}}\right)$$
- **Intuitive Explanation:** Dynamically scales weight regularization according to the layer's position in the Lucas-Pell harmonic sequence, preserving essential feature manifolds while aggressively pruning noisy weights.
- **Usage Example:**
  ```python
  from nrc_ai import LucasPellHybridWeightDecay

  decay = LucasPellHybridWeightDecay(base_decay=1e-4)
  penalty = decay(model.parameters())
  ```

### 3.5 MST Lyapunov Clipping (`MSTLyapunovClipping`)
- **Mathematical Anchor:** 
  $$\tilde{g} = g \cdot \min\left(1, \frac{\lambda_{\text{max}}}{\|g\|_2 + \phi^{-2}}\right)$$
- **Intuitive Explanation:** Prevents gradient explosion by strictly bounding update scales to the maximum Lyapunov exponent of the underlying state manifold.
- **Usage Example:**
  ```python
  from nrc_ai import MSTLyapunovClipping

  clipper = MSTLyapunovClipping(max_lyapunov=1.0)
  clipper.clip_gradients(model)
  ```

### 3.6 Biological Exclusion Gradient Router (`BiologicalExclusionGradientRouter`)
- **Mathematical Anchor:** Custom autograd function filtering out backward gradients whose modular indices fall into the chaotic void $\{0, 3, 6, 9\} \pmod 9$.
- **Intuitive Explanation:** Directs backpropagation away from non-resonant parameter updates, protecting critical feature representations from corrupting local updates.
- **Usage Example:**
  ```python
  from nrc_ai import BiologicalExclusionGradientRouter

  router = BiologicalExclusionGradientRouter()
  filtered_grad = router(raw_features)
  ```

### 3.7 GTT Entropy Collapse & Navier-Stokes Regularizers (`GTTEntropyCollapseRegularizer`, `NavierStokesDampingRegularizer`)
- **Mathematical Anchor:** Fluid-dynamic damping terms and cross-layer entropy bounds preventing internal covariate drift and representation collapse.
- **Usage Example:**
  ```python
  from nrc_ai import GTTEntropyCollapseRegularizer, NavierStokesDampingRegularizer

  gtt_reg = GTTEntropyCollapseRegularizer(weight=0.01)
  ns_reg = NavierStokesDampingRegularizer(viscosity=0.05)
  loss = loss + gtt_reg(activations) + ns_reg(activations)
  ```

### 3.8 NRC Entropy Attractor Early Stopping (`NRCEntropyAttractorEarlyStopping`)
- **Mathematical Anchor:** Monitors validation loss convergence rates against Lyapunov stabilization thresholds, stopping training when entropy reaches theoretical resonance.
- **Usage Example:**
  ```python
  from nrc_ai import NRCEntropyAttractorEarlyStopping

  early_stopper = NRCEntropyAttractorEarlyStopping(patience=5, min_delta=1e-4)
  if early_stopper(val_loss):
      print("Optimal resonance reached. Stopping training.")
  ```

---

## 4. Neural Layers, Basis Encodings & Activations

### 4.1 E8 Golden Basis Embedding (`E8GoldenBasisEmbedding`)
- **Mathematical Anchor:** Projects continuous input vectors onto the roots of the $E_8$ exceptional Lie algebra scaled by powers of $\phi$.
- **Intuitive Explanation:** Replaces standard random lookup embeddings with regular lattice anchors that maximize information density per dimension and preserve geometric symmetries.
- **Usage Example:**
  ```python
  from nrc_ai import E8GoldenBasisEmbedding

  e8_emb = E8GoldenBasisEmbedding(num_embeddings=1000, embedding_dim=256)
  tokens = torch.randint(0, 1000, (2, 32))
  vectors = e8_emb(tokens)
  ```

### 4.2 Floor-Sinh Activation (`FloorSinhActivation`)
- **Mathematical Anchor:** 
  $$f(x) = \frac{\lfloor \sinh(x \cdot \phi) \rfloor}{\phi}$$
- **Intuitive Explanation:** A discrete-continuous non-linear activation that quantizes signals onto deterministic resonant energy bands, preventing gradual activation drift in ultra-deep networks.
- **Usage Example:**
  ```python
  from nrc_ai import FloorSinhActivation

  act = FloorSinhActivation()
  y = act(torch.randn(4, 64))
  ```

### 4.3 Golden Flow Normalization (`GoldenFlowNorm`)
- **Mathematical Anchor:** 
  $$y = \frac{x}{\|x\|_2 + \phi^{-2}} \odot \gamma + \beta$$
- **Intuitive Explanation:** A fast, stable alternative to LayerNorm and RMSNorm that scales feature vector norms along the optimal golden flow vector path, guaranteeing non-zero bounded denominators without artificial $\epsilon$ tuning.
- **Usage Example:**
  ```python
  from nrc_ai import GoldenFlowNorm

  norm = GoldenFlowNorm(normalized_shape=512)
  normalized = norm(torch.randn(2, 64, 512))
  ```

### 4.4 Phi-Infinity Lossless LoRA (`PhiInfinityLosslessLoRA`)
- **Mathematical Anchor:** 
  $$\Delta \mathbf{W} = \left(\mathbf{A} \otimes \mathbf{B}\right) \cdot \phi^{-r/2}$$
- **Intuitive Explanation:** Enhances Low-Rank Adaptation (LoRA) by structuring adapter matrices $\mathbf{A}$ and $\mathbf{B}$ along self-similar fractal dimensions, allowing parameter-efficient fine-tuning without losing high-frequency domain knowledge.
- **Usage Example:**
  ```python
  from nrc_ai import PhiInfinityLosslessLoRA

  lora_layer = PhiInfinityLosslessLoRA(in_features=512, out_features=512, rank=8)
  y = lora_layer(torch.randn(4, 512))
  ```

### 4.5 Triple Theta Initializer (`TripleThetaInitializer`)
- **Mathematical Anchor:** Initializes linear weights using deterministic triple-theta coordinate rotations derived from the Jacobi theta functions.
- **Intuitive Explanation:** Eliminates random initialization variance (Xavier/Kaiming randomness), ensuring every layer begins at an optimal spectral radius for smooth gradient propagation from step zero.
- **Usage Example:**
  ```python
  from nrc_ai import TripleThetaInitializer

  linear = TripleThetaInitializer(in_features=256, out_features=256)
  ```

### 4.6 Prime Density Generation & TUPT Token Pruning (`PrimeDensityConditionedGeneration`, `TUPTExclusionTokenPruning`)
- **Mathematical Anchor:** Token logit distribution sampling conditioned on prime density distributions, paired with dynamic pruning of low-entropy sequence tokens.
- **Usage Example:**
  ```python
  from nrc_ai import PrimeDensityConditionedGeneration, TUPTExclusionTokenPruning

  sampler = PrimeDensityConditionedGeneration(vocab_size=32000)
  pruner = TUPTExclusionTokenPruning(keep_ratio=0.7)
  ```

### 4.7 Multi-Manifold Cross-Domain Primitives (`GeometricLatticeIsomorphism`, `NRCProteinFoldingEngine`, `TUPTSyncSeed`)
- **Mathematical Anchor:** Transforms high-dimensional representations between physical metamaterials, macromolecular biophysics, and LLM latent vectors.
- **Usage Example:**
  ```python
  from nrc_ai import GeometricLatticeIsomorphism, NRCProteinFoldingEngine

  isomorphism = GeometricLatticeIsomorphism(in_dim=256, out_dim=729)
  bio_engine = NRCProteinFoldingEngine(d_model=256)
  ```

---

## Installation & Quickstart

### Setup via `uv` (Recommended)

```bash
# Clone the repository
git clone https://github.com/Nexus-Resonance-Codex/Ai-Enhancements.git
cd Ai-Enhancements

# Create and activate virtual environment with uv
uv venv
source .venv/bin/activate

# Install in editable mode with development dependencies
uv pip install -e ".[dev]"
```

Alternatively, install via standard `pip`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

---

## Complete Drop-in Transformer Example

Below is a complete, runnable example assembling multiple NRC AI enhancements into a self-stabilizing Transformer layer:

```python
import torch
import torch.nn as nn
from nrc_ai import (
    GoldenFlowNorm,
    GoldenSpiralRotaryEmbedding,
    HodgePhiTTorsionAttention,
    PhiInfinityLosslessLoRA,
)

class ResonantTransformerBlock(nn.Module):
    def __init__(self, dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.norm1 = GoldenFlowNorm(dim)
        self.attn = HodgePhiTTorsionAttention(embed_dim=dim, num_heads=num_heads)
        self.rope = GoldenSpiralRotaryEmbedding(dim=dim // num_heads)
        
        self.norm2 = GoldenFlowNorm(dim)
        self.ffn = nn.Sequential(
            PhiInfinityLosslessLoRA(in_features=dim, out_features=dim * 4, rank=16),
            nn.GELU(),
            PhiInfinityLosslessLoRA(in_features=dim * 4, out_features=dim, rank=16),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm + Hodge Torsion Attention
        h = self.norm1(x)
        attn_out, _ = self.attn(h)
        x = x + attn_out
        
        # Pre-norm + Lossless LoRA Feed-Forward
        x = x + self.ffn(self.norm2(x))
        return x

# Instantiate and test
model = ResonantTransformerBlock(dim=512, num_heads=8)
x = torch.randn(2, 64, 512)
out = model(x)
print("Resonant Block Output:", out.shape)  # torch.Size([2, 64, 512])
```

---

## Verification & Test Execution

Run the complete test suite to verify mathematical precision and tensor bounds across all 30+ modules:

```bash
# Execute unit test suite (68 tests)
pytest tests/ -v

# Run format and lint checks
ruff check src/ tests/
ruff format --check src/ tests/
```

---

## Licensing & Governance

The Nexus Resonance Codex operates under an institutional Dual-License model:

- **Open Source and Academic Research:**
  - Codebase: [GNU Affero General Public License v3.0 (AGPL-3.0)](LICENSE)
  - Data & Weights: [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0)](LICENSE-DATA)
  - Patent Covenant: [Tesla-Style Patent Pledge](PATENT_PLEDGE.md)
  - Trademark: [Trademark and Nomenclature Policy](TRADEMARK_POLICY.md)

- **Enterprise & Commercial Use:**
  Commercial organizations requiring closed-source, proprietary deployment without AGPL-3.0 copyleft terms must obtain a commercial license. See [COMMERCIAL_USE.md](COMMERCIAL_USE.md) or contact:

  **James Paul Trageser**  
  Founder and Chief Architect  
  Email: `NexusResonanceCodex@gmail.com`

---

## Academic Citation

```bibtex
@software{trageser2026nrc_ai,
  author       = {James Paul Trageser},
  title        = {Nexus Resonance Codex (NRC): Cognitive Resonance and Deterministic AI Enhancements},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub Repository},
  howpublished = {\url{https://github.com/Nexus-Resonance-Codex/Ai-Enhancements}}
}
```

---

*Copyright (c) 2026 Nexus Resonance Codex (NRC). All rights reserved.*
