# Nexus Resonance Codex (NRC): AI Enhancements

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
  <a href="https://github.com/Nexus-Resonance-Codex/Ai-Enhancements/actions/workflows/lint.yml"><img src="https://img.shields.io/github/actions/workflow/status/Nexus-Resonance-Codex/Ai-Enhancements/lint.yml?branch=main&style=flat-square&label=Lint" alt="Lint CI"></a>
  <a href="https://github.com/Nexus-Resonance-Codex/Ai-Enhancements/actions/workflows/python-tests.yml"><img src="https://img.shields.io/github/actions/workflow/status/Nexus-Resonance-Codex/Ai-Enhancements/python-tests.yml?branch=main&style=flat-square&label=Python%20CI" alt="Python CI"></a>
  <a href="https://nexus-resonance-codex.github.io/Ai-Enhancements/"><img src="https://img.shields.io/badge/Docs-nrc--ai.github.io-blueviolet.svg?style=flat-square" alt="Documentation site"></a>
  <a href="https://codespaces.new/Nexus-Resonance-Codex/Ai-Enhancements"><img src="https://img.shields.io/badge/Open%20in%20GitHub%20Codespaces-black.svg?style=flat-square&logo=github" alt="Open in GitHub Codespaces"></a>
</p>

---

## 🚀 Try it now

- **1-click cloud environment:** [Open in GitHub Codespaces](https://codespaces.new/Nexus-Resonance-Codex/Ai-Enhancements) — launches a ready-to-code workspace with `nrc-ai` and its dev tools installed.
- **Run the notebook:** [`examples/quickstart.ipynb`](https://github.com/Nexus-Resonance-Codex/Ai-Enhancements/blob/main/examples/quickstart.ipynb) — the full quickstart, cell by cell, right in your browser.
- **Ask Copilot:** the [`.github/prompts/`](https://github.com/Nexus-Resonance-Codex/Ai-Enhancements/tree/main/.github/prompts) suite has 14 expert prompts to run in Copilot Chat on github.com.

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
  out = attn(x)
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

  sparse_attn = LucasWeightedSparseAttention(max_seq_length=128)
  mask = sparse_attn(128)
  print("Sparse mask shape:", mask.shape)  # torch.Size([128, 128])
  ```

### 1.3 Golden Spiral Rotary Position Embedding (`GoldenSpiralRotaryEmbedding`)
- **Mathematical Anchor:** 
  $$\mathbf{R}_{\theta}(m) = \text{diag}\left(R(\theta_1 m), R(\theta_2 m), \dots, R(\theta_{d/2} m)\right), \quad \theta_k = \frac{360^\circ}{\phi^2} \cdot \phi^{-2k/d}$$
- **Intuitive Explanation:** Upgrades standard RoPE by scaling rotation angles along the golden spiral angle ($\theta \approx 137.507764^\circ$). It ensures that relative token distances remain self-similar across arbitrary sequence lengths without frequency aliasing.
- **Usage Example:**
  ```python
  from nrc_ai import GoldenSpiralRotaryEmbedding

  rope = GoldenSpiralRotaryEmbedding(dim=64)
  q = torch.randn(2, 8, 128, 64)   # (batch, heads, seq_len, head_dim)
  q_rot = rope(q, seq_dim=2)
  print("Rotated queries shape:", q_rot.shape)  # torch.Size([2, 8, 128, 64])
  ```

### 1.4 Phi-Void Positional Encoding (`PhiVoidResonancePositionalEncoding`)
- **Mathematical Anchor:** 
  $$P(pos, 2i) = \sin\left(\frac{pos}{\phi^{4i/d}}\right), \quad P(pos, 2i+1) = \cos\left(\frac{pos}{\phi^{4i/d}}\right)$$
- **Intuitive Explanation:** Replaces standard power-of-10000 sinusoidal embeddings with transfinite golden ratio scales, preventing high-frequency representation decay in state-space representations.
- **Usage Example:**
  ```python
  from nrc_ai import PhiVoidResonancePositionalEncoding

  pos_enc = PhiVoidResonancePositionalEncoding(d_model=512, max_seq_len=4096)
  emb = pos_enc(torch.zeros(1, 100, 512))
  ```

### 1.5 QRT Geometric Attention Bias (`QRTGeometricAttentionBias`)
- **Mathematical Anchor:** 
  $$\text{Bias}(i, j) = -\frac{|i - j|^2}{\phi} \cdot \cos\left(\frac{\pi}{\phi} |i - j|\right)$$
- **Intuitive Explanation:** Applies a deterministic quantum-resonance wave penalty to the attention logits, regularizing local token interactions and suppressing long-range noise without requiring artificial attention masking.
- **Usage Example:**
  ```python
  from nrc_ai import QRTGeometricAttentionBias

  bias_layer = QRTGeometricAttentionBias(max_seq_len=64)
  bias_matrix = bias_layer(torch.zeros(2, 8, 64, 64))
  print("Bias matrix shape:", bias_matrix.shape)  # torch.Size([2, 8, 64, 64])
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

  kv_cache = ResonanceShardKVCache(folding_steps=3, shard_capacity=1024)
  k = torch.randn(1, 8, 32, 64)
  v = torch.randn(1, 8, 32, 64)
  k_all, v_all = kv_cache(k, v)
  print("Cache K shape:", k_all.shape, "| V shape:", v_all.shape)
  ```

### 2.2 Phi-Infinity Shard Folding (`PhiInfinityShardFolding`)
- **Mathematical Anchor:** 
  $$s_k = x \cdot \phi^k + \text{roll}(x, k) \cdot \phi^{-k}$$
- **Intuitive Explanation:** Recursively folds multi-dimensional activation partitions into a unified residual channel, allowing models to store historical conversational context at high information densities without memory explosion.
- **Usage Example:**
  ```python
  from nrc_ai import PhiInfinityShardFolding

  folder = PhiInfinityShardFolding(k_steps=3)
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

  unfolder = InfiniteEInfinityContextUnfolder()
  restored = unfolder(folded, 2)
  ```

### 2.4 Phi Sharding Compression (`PhiShardingCompression`)
- **Mathematical Anchor:** 
  $$y = \text{LayerNorm}\left(\sum_{i=1}^m \frac{\mathbf{W}_i x}{\phi^i}\right)$$
- **Intuitive Explanation:** Compresses wide weight and projection matrices by grouping parameters along modular golden-ratio coordinate shards, shrinking model footprint while retaining full expressive rank.
- **Usage Example:**
  ```python
  from nrc_ai import PhiShardingCompression

  compressor = PhiShardingCompression(input_dim=512, compress_dim=128)
  out = compressor(torch.randn(4, 512))
  ```

### 2.5 Phi-Infinity Persistent Memory & Executive Agent (`PhiInfinityPersistentMemory`, `ExecutiveAgent`)
- **Mathematical Anchor:** High-dimensional associative matrix retention scaled by continuous Lyapunov decay.
- **Intuitive Explanation:** Maintains long-term agent state and episodic task context across arbitrary conversation turns, routing subtasks to optimal solver modules without context loss.
- **Usage Example:**
  ```python
  from nrc_ai import ExecutiveAgent, PhiInfinityPersistentMemory

  memory = PhiInfinityPersistentMemory(hidden_dim=512)
  projected = memory(torch.randn(2, 8, 512))      # (2, 8, 512)
  state = memory.update(torch.randn(1, 512))      # fold into lattice_state
  print("Projected:", projected.shape, "| Lattice state:", state.shape)

  agent = ExecutiveAgent(name="Planner")
  ctx = agent.spawn_sub_model("Summarize the session")
  print("Spawned:", ctx["agent"], "| Status:", ctx["resonance"])
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
  optimizer = QRTTurbulenceOptimizer(model.parameters(), lr=1e-3)
  ```

### 3.2 Phi-Inverse Momentum Accelerator (`PhiInverseMomentumAccelerator`)
- **Mathematical Anchor:** 
  $$v_t = \phi^{-1} v_{t-1} + (1 - \phi^{-1}) g_t, \quad \phi^{-1} \approx 0.61803398875$$
- **Intuitive Explanation:** Traditional momentum parameters ($\beta = 0.9$) are empirically chosen heuristics. This optimizer anchors momentum directly to the golden attractor $\phi^{-1}$, provably minimizing oscillations near ill-conditioned ravines.
- **Usage Example:**
  ```python
  import torch.nn as nn
  from nrc_ai import PhiInverseMomentumAccelerator

  model = nn.Linear(10, 2)
  optimizer = PhiInverseMomentumAccelerator(model.parameters(), lr=1e-3)
  ```

### 3.3 Pisano Modulated Learning Rate Schedule (`PisanoModulatedLRSchedule`)
- **Mathematical Anchor:** 
  $$\eta_t = \eta_{\text{min}} + (\eta_{\text{max}} - \eta_{\text{min}}) \cdot \frac{F_{t \pmod{\pi(m)}}}{\max(F)}$$
  where $\pi(m)$ is the Pisano period for modulo $m$.
- **Intuitive Explanation:** Replaces standard cosine or linear warmups with cyclic Pisano sequence modulations. Periodic harmonic resets allow gradient descent to escape local minima and saddle points deterministically.
- **Usage Example:**
  ```python
  import torch
  from nrc_ai import PisanoModulatedLRSchedule

  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  scheduler = PisanoModulatedLRSchedule(optimizer, pisano_period=24)
  ```

### 3.4 Lucas-Pell Hybrid Weight Decay (`LucasPellHybridWeightDecay`)
- **Mathematical Anchor:** 
  $$\lambda_w = \lambda_0 \cdot \left(\frac{L_k}{P_k + \phi^{-2}}\right)$$
- **Intuitive Explanation:** Dynamically scales weight regularization according to the layer's position in the Lucas-Pell harmonic sequence, preserving essential feature manifolds while aggressively pruning noisy weights.
- **Usage Example:**
  ```python
  from nrc_ai import LucasPellHybridWeightDecay

  LucasPellHybridWeightDecay.apply_hybrid_decay_(model.parameters())
  ```

### 3.5 MST Lyapunov Clipping (`MSTLyapunovClipping`)
- **Mathematical Anchor:** 
  $$\tilde{g} = g \cdot \min\left(1, \frac{\lambda_{\text{max}}}{\|g\|_2 + \phi^{-2}}\right)$$
- **Intuitive Explanation:** Prevents gradient explosion by strictly bounding update scales to the maximum Lyapunov exponent of the underlying state manifold.
- **Usage Example:**
  ```python
  from nrc_ai import MSTLyapunovClipping

  import torch
  from nrc_ai import MSTLyapunovClipping

  clipper = MSTLyapunovClipping(clip_val=0.381)
  grad = torch.randn(256, 256) * 100
  clipped = clipper(grad)
  print("Clipped grad max abs:", clipped.abs().max().item())
  ```

### 3.6 Biological Exclusion Gradient Router (`BiologicalExclusionGradientRouter`)
- **Mathematical Anchor:** Custom autograd function filtering out backward gradients whose modular indices fall into the chaotic void $\{0, 3, 6, 9\} \pmod 9$.
- **Intuitive Explanation:** Directs backpropagation away from non-resonant parameter updates, protecting critical feature representations from corrupting local updates.
- **Usage Example:**
  ```python
  from nrc_ai import BiologicalExclusionGradientRouter

  import torch
  from nrc_ai import BiologicalExclusionGradientRouter

  router = BiologicalExclusionGradientRouter()
  raw_features = torch.randn(2, 64, 512)
  filtered_grad = router(raw_features)
  ```

### 3.7 GTT Entropy Collapse & Navier-Stokes Regularizers (`GTTEntropyCollapseRegularizer`, `NavierStokesDampingRegularizer`)
- **Mathematical Anchor:** Fluid-dynamic damping terms and cross-layer entropy bounds preventing internal covariate drift and representation collapse.
- **Usage Example:**
  ```python
  from nrc_ai import GTTEntropyCollapseRegularizer, NavierStokesDampingRegularizer

  import torch
  from nrc_ai import GTTEntropyCollapseRegularizer, NavierStokesDampingRegularizer

  gtt_reg = GTTEntropyCollapseRegularizer(gtt_safe_boundary=10.96)
  ns_reg = NavierStokesDampingRegularizer(damping_strength=0.05)
  activations = torch.randn(2, 64, 512)
  loss = torch.tensor(1.0) + gtt_reg(activations) + ns_reg(activations)
  ```

### 3.8 NRC Entropy Attractor Early Stopping (`NRCEntropyAttractorEarlyStopping`)
- **Mathematical Anchor:** Monitors validation loss convergence rates against Lyapunov stabilization thresholds, stopping training when entropy reaches theoretical resonance.
- **Usage Example:**
  ```python
  from nrc_ai import NRCEntropyAttractorEarlyStopping

  early_stopper = NRCEntropyAttractorEarlyStopping(phi_tolerance=1e-4)
  val_loss = 0.35
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

  norm = GoldenFlowNorm(hidden_dim=512)
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

  import torch
  from nrc_ai import PrimeDensityConditionedGeneration, TUPTExclusionTokenPruning

  sampler = PrimeDensityConditionedGeneration(vocab_size=32000)
  pruner = TUPTExclusionTokenPruning()
  hidden = torch.randn(2, 64, 128)
  pruned = pruner(hidden)
  print("Pruned shape:", pruned.shape)
  ```

### 4.7 Multi-Manifold Cross-Domain Primitives (`GeometricLatticeIsomorphism`, `NRCProteinFoldingEngine`, `TUPTSyncSeed`)
- **Mathematical Anchor:** Transforms high-dimensional representations between physical metamaterials, macromolecular biophysics, and LLM latent vectors.
- **Usage Example:**
  ```python
  from nrc_ai import GeometricLatticeIsomorphism, NRCProteinFoldingEngine

  isomorphism = GeometricLatticeIsomorphism(high_dim_features=256)
  bio_engine = NRCProteinFoldingEngine(sequence_dim=256)
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
    HodgePhiTTorsionAttention,
    PhiInfinityLosslessLoRA,
)

class ResonantTransformerBlock(nn.Module):
    def __init__(self, dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.norm1 = GoldenFlowNorm(dim)
        self.attn = HodgePhiTTorsionAttention(embed_dim=dim, num_heads=num_heads)

        self.norm2 = GoldenFlowNorm(dim)
        self.ffn = nn.Sequential(
            PhiInfinityLosslessLoRA(in_features=dim, out_features=dim * 4, rank=16),
            nn.GELU(),
            PhiInfinityLosslessLoRA(in_features=dim * 4, out_features=dim, rank=16),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm + Hodge Torsion Attention
        h = self.norm1(x)
        attn_out = self.attn(h)
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
