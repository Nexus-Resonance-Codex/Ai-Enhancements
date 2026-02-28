#!/usr/bin/env python3
"""
=======================================================================
  NRC AI Enhancement Suite — Full Demo (All 30 Enhancements)
=======================================================================
  Author:   James Trageser (@jtrag)
  Repo:     https://github.com/Nexus-Resonance-Codex/ai-enhancements
  License:  NRC-L v2.0

  This script instantiates every single one of the 30 NRC AI
  Enhancements on synthetic data, runs a forward pass, and verifies
  that each module produces valid (no NaN/Inf) output.

  HOW TO RUN:
    cd ai-enhancements
    pip install -e .
    python examples/demo_all_enhancements.py
=======================================================================
"""
import sys
import os
import torch
import math

# ---------------------------------------------------------------------------
# Path setup — allow running from repo root without installing
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from nrc_math.phi import PHI_FLOAT, PHI_INVERSE_FLOAT, binet_formula
from nrc_math.qrt import qrt_damping, execute_qrt_damping_tensor
from nrc_math.mst import mst_step
from nrc_math.tupt_exclusion import apply_exclusion_gate

from enhancements import (
    PhiInfinityShardFolding,
    NRCProteinFoldingEngine,
    GoldenAttractorFlowNorm,
    TripleThetaInitialiser,
    ResonanceShardKVCache,
    BiologicalExclusionGradientRouter,
    HodgePhiTTorsionAttention,
    E8GoldenBasisEmbedding,
    PhiInfinityLosslessLoRA,
    NavierStokesDampingRegulariser,
    PrimeDensityConditionedGeneration,
    GTTEntropyCollapseRegulariser,
    PhiInverseMomentumAccelerator,
    TUPTAttractorSyncSeed,
    QRTKernelConvolution,
    LucasWeightedSparseAttentionMask,
    PhiResonantWeighting,
    GizaLatticeIsomorphismProjection,
    MSTLyapunovGradientClipper,
    PisanoModulatedLRSchedule,
    LucasPellHybridWeightDecay,
    TUPTExclusionTokenPruner,
    PhiVoidResonancePositionalEncoding,
    InfiniteContextShardUnfolder,
    ModularDropout3697,
    QRTTurbulenceAdaptiveOptimizer,
    GizaSlopeAngleAwareAttentionBias,
    FloorSinhActivationRegularizer,
    GoldenSpiralRotaryEmbedding,
    NRCEntropyAttractorEarlyStopping,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def check_tensor(name: str, t: torch.Tensor):
    """Validates that a tensor has no NaN or Inf values."""
    assert not torch.isnan(t).any(), f"  ✗ {name} — NaN detected!"
    assert not torch.isinf(t).any(), f"  ✗ {name} — Inf detected!"
    print(f"  ✓ {name:.<55s} shape={list(t.shape)}")


# ---------------------------------------------------------------------------
# Constants for synthetic data
# ---------------------------------------------------------------------------
BATCH     = 2
SEQ_LEN   = 64
EMBED_DIM = 128
NUM_HEADS = 4
HEAD_DIM  = EMBED_DIM // NUM_HEADS
VOCAB     = 1000


def main():
    print("=" * 72)
    print("  NEXUS RESONANCE CODEX — AI Enhancement Suite Demo")
    print(f"  φ  = {PHI_FLOAT:.15f}")
    print(f"  φ⁻¹= {PHI_INVERSE_FLOAT:.15f}")
    print(f"  F₁₀= {binet_formula(10):.1f}  (Binet check)")
    print("=" * 72)

    # Shared synthetic tensors
    x = torch.randn(BATCH, SEQ_LEN, EMBED_DIM)
    x_small = torch.randn(BATCH, SEQ_LEN, 32)
    tokens = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))

    # ------------------------------------------------------------------
    print("\n── NRC Math Foundation ──────────────────────────────────────")
    # ------------------------------------------------------------------
    qrt_out = qrt_damping(x)
    check_tensor("QRT Wave Function", qrt_out)

    qrt_out2 = execute_qrt_damping_tensor(x)
    check_tensor("QRT Execute Tensor (alias)", qrt_out2)

    mst_out = mst_step(torch.tensor([0.1, 0.5, 1.0, 2.0]))
    check_tensor("MST Step Function", mst_out)

    gate_out = apply_exclusion_gate(torch.arange(20, dtype=torch.float32))
    check_tensor("TUPT Exclusion Gate", gate_out)

    # ------------------------------------------------------------------
    print("\n── Enhancements 1-10 (Core Architecture) ───────────────────")
    # ------------------------------------------------------------------

    # 1. Shard Folding
    e1 = PhiInfinityShardFolding(k_steps=3)
    check_tensor("#01 Shard Folding", e1(x))

    # 2. Protein Engine
    e2 = NRCProteinFoldingEngine()
    check_tensor("#02 Protein Engine", e2(x))

    # 3. GAFEN
    e3 = GoldenAttractorFlowNorm(normalized_shape=EMBED_DIM)
    check_tensor("#03 GAFEN", e3(x))

    # 4. Triple-Theta Init
    e4 = TripleThetaInitialiser()
    w = torch.empty(EMBED_DIM, EMBED_DIM)
    e4.initialise(w)
    check_tensor("#04 Triple-Theta Init", w)

    # 5. Resonance KV Cache
    e5 = ResonanceShardKVCache(head_dim=HEAD_DIM)
    k_cache = torch.randn(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    v_cache = torch.randn(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    k_out, v_out = e5(k_cache, v_cache)
    check_tensor("#05 Resonance KV Cache (K)", k_out)
    check_tensor("#05 Resonance KV Cache (V)", v_out)

    # 6. Exclusion Gradient Router
    e6 = BiologicalExclusionGradientRouter()
    check_tensor("#06 Exclusion Gradient Router", e6(x))

    # 7. Hodge Torsion Attention
    e7 = HodgePhiTTorsionAttention(embed_dim=EMBED_DIM, num_heads=NUM_HEADS)
    check_tensor("#07 Hodge Torsion Attention", e7(x))

    # 8. E8 Golden Basis Embedding
    e8 = E8GoldenBasisEmbedding(vocab_size=VOCAB, embed_dim=EMBED_DIM)
    check_tensor("#08 E8 Golden Basis", e8(tokens))

    # 9. Phi Lossless LoRA
    e9 = PhiInfinityLosslessLoRA(in_features=EMBED_DIM, out_features=EMBED_DIM, rank=16)
    check_tensor("#09 Phi LoRA Adapter", e9(x))

    # 10. Navier-Stokes Damping
    e10 = NavierStokesDampingRegulariser(damping_strength=0.01)
    check_tensor("#10 Navier-Stokes Damping", e10(x))

    # ------------------------------------------------------------------
    print("\n── Enhancements 11-20 (Generation & Stability) ─────────────")
    # ------------------------------------------------------------------

    # 11. Prime Density Generation
    e11 = PrimeDensityConditionedGeneration(vocab_size=VOCAB)
    logits_in = torch.randn(BATCH, SEQ_LEN, VOCAB)
    check_tensor("#11 Prime Density Gen", e11(logits_in))

    # 12. GTT Entropy Regulariser
    e12 = GTTEntropyCollapseRegulariser()
    check_tensor("#12 GTT Entropy Collapse", e12(x))

    # 13. Phi Momentum Accelerator (optimizer — step test)
    dummy_model = torch.nn.Linear(EMBED_DIM, EMBED_DIM)
    e13 = PhiInverseMomentumAccelerator(dummy_model.parameters(), lr=1e-3)
    loss = dummy_model(x.mean(dim=1)).sum()
    loss.backward()
    e13.step()
    print(f"  ✓ {'#13 Phi Momentum Accelerator':.<55s} optimizer step OK")

    # 14. TUPT Sync Seed
    e14 = TUPTAttractorSyncSeed()
    e14.seed()
    print(f"  ✓ {'#14 TUPT Sync Seed':.<55s} deterministic seed set")

    # 15. QRT Convolution
    e15 = QRTKernelConvolution(in_channels=EMBED_DIM, out_channels=EMBED_DIM)
    conv_in = torch.randn(BATCH, EMBED_DIM, SEQ_LEN)
    check_tensor("#15 QRT Convolution", e15(conv_in))

    # 16. Lucas Sparse Attention Mask
    e16 = LucasWeightedSparseAttentionMask(seq_len=SEQ_LEN)
    mask = e16()
    check_tensor("#16 Lucas Sparse Mask", mask)

    # 17. Phi Resonant Weighting
    e17 = PhiResonantWeighting()
    check_tensor("#17 Phi Resonant Weighting", e17(x))

    # 18. Giza Isomorphism
    e18 = GizaLatticeIsomorphismProjection(dim=EMBED_DIM)
    check_tensor("#18 Giza Isomorphism", e18(x))

    # 19. MST Lyapunov Clipping
    e19 = MSTLyapunovGradientClipper()
    grad = torch.randn(EMBED_DIM, EMBED_DIM) * 100
    check_tensor("#19 MST Lyapunov Clip", e19(grad))

    # 20. Pisano LR Schedule
    opt20 = torch.optim.SGD(dummy_model.parameters(), lr=1e-3)
    e20 = PisanoModulatedLRSchedule(opt20, pisano_period=24)
    for _ in range(24):
        e20.step()
    print(f"  ✓ {'#20 Pisano LR Schedule':.<55s} cycled 24 steps, lr={e20.get_last_lr()[0]:.6f}")

    # ------------------------------------------------------------------
    print("\n── Enhancements 21-30 (Automation & Boundaries) ─────────────")
    # ------------------------------------------------------------------

    # 21. Lucas-Pell Weight Decay
    e21 = LucasPellHybridWeightDecay(dummy_model.parameters(), lr=1e-3)
    loss2 = dummy_model(x.mean(dim=1)).sum()
    loss2.backward()
    e21.step()
    print(f"  ✓ {'#21 Lucas-Pell Weight Decay':.<55s} optimizer step OK")

    # 22. TUPT Token Pruning
    e22 = TUPTExclusionTokenPruner()
    check_tensor("#22 TUPT Token Pruning", e22(x))

    # 23. Phi Void Positional Encoding
    e23 = PhiVoidResonancePositionalEncoding(dim=EMBED_DIM)
    check_tensor("#23 Phi Void Positional", e23(x))

    # 24. Infinite Context Shard Unfolder
    e24 = InfiniteContextShardUnfolder()
    check_tensor("#24 Shard Unfolder", e24(x))

    # 25. 3-6-9-7 Modular Dropout
    e25 = ModularDropout3697()
    check_tensor("#25 Modular Dropout", e25(x))

    # 26. QRT Turbulence Optimizer
    e26 = QRTTurbulenceAdaptiveOptimizer(dummy_model.parameters(), lr=1e-3)
    loss3 = dummy_model(x.mean(dim=1)).sum()
    loss3.backward()
    e26.step()
    print(f"  ✓ {'#26 QRT Turbulence Optimizer':.<55s} optimizer step OK")

    # 27. Giza Slope Attention Bias
    e27 = GizaSlopeAngleAwareAttentionBias(seq_len=SEQ_LEN)
    check_tensor("#27 Giza Attention Bias", e27())

    # 28. Floor-Sinh Activation
    e28 = FloorSinhActivationRegularizer()
    check_tensor("#28 Floor-Sinh Activation", e28(x_small))

    # 29. Golden Spiral RoPE
    e29 = GoldenSpiralRotaryEmbedding(dim=EMBED_DIM, max_seq_len=SEQ_LEN)
    check_tensor("#29 Golden Spiral RoPE", e29(x))

    # 30. Entropy-Attractor Early Stopping
    e30 = NRCEntropyAttractorEarlyStopping(phi_tolerance=1e-4)
    stop1 = e30(10.0)
    stop2 = e30(6.18)
    print(f"  ✓ {'#30 Entropy Early Stopping':.<55s} stop={stop1}, {stop2}")

    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  ALL 30 ENHANCEMENTS VALIDATED SUCCESSFULLY")
    print("  Nexus Resonance Online. Systems Calibrated to Phi.")
    print("=" * 72)


if __name__ == "__main__":
    main()
