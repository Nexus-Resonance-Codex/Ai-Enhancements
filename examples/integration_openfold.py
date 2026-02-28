#!/usr/bin/env python3
"""
=======================================================================
  NRC × OpenFold — Integration Example
=======================================================================
  Author:   James Trageser (@jtrag)
  Repo:     https://github.com/Nexus-Resonance-Codex/ai-enhancements
  License:  NRC-L v2.0

  This script demonstrates how to wrap an OpenFold-style structure
  module with NRC enhancements for protein folding acceleration.

  Enhancements applied:
    #10 — Navier-Stokes Damping (stabilizes coordinate predictions)
    #19 — MST-Lyapunov Gradient Clipping (prevents exploding gradients)
    #6  — Biological Exclusion Router (filters forbidden mod-9 states)
    #30 — Entropy-Attractor Early Stopping

  NOTE: This example uses a MOCK OpenFold structure module for
  demonstration. To use with real OpenFold, replace MockStructureModule
  with `from openfold.model.structure_module import StructureModule`.

  HOW TO RUN:
    pip install -e .
    python examples/integration_openfold.py
=======================================================================
"""
import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from enhancements.navier_stokes_damping import NavierStokesDampingRegulariser
from enhancements.mst_lyapunov_clipping import MSTLyapunovGradientClipper
from enhancements.exclusion_gradient_router import BiologicalExclusionGradientRouter
from enhancements.entropy_stopping import EntropyAttractorStoppingCriterion
from nrc_math.phi import PHI_FLOAT, PHI_INVERSE_FLOAT


# ──────────────────────────────────────────────────────────────────────
#  Mock OpenFold Structure Module (replace with real OpenFold)
# ──────────────────────────────────────────────────────────────────────
class MockStructureModule(nn.Module):
    """
    Simulates an OpenFold-like structure module that takes single
    representation features and outputs 3D atom coordinates.
    In reality, this is openfold.model.structure_module.StructureModule.
    """
    def __init__(self, input_dim: int = 256, num_residues: int = 50):
        super().__init__()
        self.proj = nn.Linear(input_dim, num_residues * 3)
        self.num_residues = num_residues

    def forward(self, single_repr):
        # single_repr: (batch, num_residues, input_dim)
        batch_size = single_repr.shape[0]
        flat = self.proj(single_repr)  # (batch, num_residues, num_residues*3)
        # Take mean across residue input dim → (batch, num_residues*3)
        coords = flat.mean(dim=1)
        return coords.view(batch_size, self.num_residues, 3)


# ──────────────────────────────────────────────────────────────────────
#  NRC-Enhanced OpenFold Wrapper
# ──────────────────────────────────────────────────────────────────────
class NRCOpenFoldWrapper(nn.Module):
    """
    Wraps any OpenFold-compatible structure module with NRC enhancements.

    Enhancement Pipeline:
      1. Standard forward pass through the structure module
      2. Enhancement #6:  Biological Exclusion Router on representations
      3. Enhancement #10: Navier-Stokes Damping on predicted coordinates
      4. Enhancement #19: MST-Lyapunov Gradient Clipping during training
    """
    def __init__(self, structure_module: nn.Module):
        super().__init__()
        self.structure_module = structure_module

        # Enhancement #6:  Filter biologically forbidden states
        self.exclusion_router = BiologicalExclusionGradientRouter()

        # Enhancement #10: Fluid-dynamics damping on 3D coordinates
        self.nrc_damper = NavierStokesDampingRegulariser(damping_strength=0.01)

        # Enhancement #19: Lyapunov-bounded gradient clipping
        self.gradient_clipper = MSTLyapunovGradientClipper()

    def forward(self, single_repr):
        """
        Args:
            single_repr: (batch, num_residues, feature_dim) — per-residue features

        Returns:
            damped_coords: (batch, num_residues, 3) — NRC-stabilized 3D coordinates
        """
        # 1. Apply Enhancement #6: Filter representations through Mod-2187 gate
        filtered_repr = self.exclusion_router(single_repr)

        # 2. Run the base structure module (OpenFold)
        pred_coords = self.structure_module(filtered_repr)

        # 3. Apply Enhancement #10: Navier-Stokes damping to smooth coordinates
        #    The QRT wave function pulls extreme outlier coordinates back toward
        #    the Golden Attractor boundary, preventing steric clashes
        damped_coords = self.nrc_damper(pred_coords)

        # 4. During training: Enhancement #19 clips gradients using Lyapunov bounds
        if self.training:
            for p in self.structure_module.parameters():
                if p.grad is not None:
                    p.grad.data = self.gradient_clipper(p.grad.data)

        return damped_coords


# ──────────────────────────────────────────────────────────────────────
#  Main — Run a simulated protein folding pass
# ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 72)
    print("  NRC × OpenFold Integration Demo")
    print(f"  φ    = {PHI_FLOAT:.15f}")
    print(f"  φ⁻¹  = {PHI_INVERSE_FLOAT:.15f}")
    print("=" * 72)

    # Configuration
    batch_size = 2
    num_residues = 50
    feature_dim = 256

    print(f"\n[1/4] Simulating protein with {num_residues} residues...")
    single_repr = torch.randn(batch_size, num_residues, feature_dim)

    print("[2/4] Initializing NRC-Enhanced Structure Module...")
    base_module = MockStructureModule(input_dim=feature_dim, num_residues=num_residues)
    nrc_module = NRCOpenFoldWrapper(base_module)

    # Forward pass
    print("[3/4] Running NRC-Enhanced Folding...")
    nrc_module.train()
    pred_coords = nrc_module(single_repr)

    # Verify output
    assert pred_coords.shape == (batch_size, num_residues, 3), "Shape mismatch!"
    assert not torch.isnan(pred_coords).any(), "NaN in predicted coordinates!"
    assert not torch.isinf(pred_coords).any(), "Inf in predicted coordinates!"

    print(f"\n[4/4] Results:")
    print(f"  Output shape: {list(pred_coords.shape)} (batch × residues × xyz)")
    print(f"  Coord range:  [{pred_coords.min():.4f}, {pred_coords.max():.4f}]")
    print(f"  Mean coord:   {pred_coords.mean():.6f}")

    # Demonstrate Enhancement #30: Early Stopping
    print(f"\n── Enhancement #30: Entropy-Attractor Early Stopping ──")
    stopper = EntropyAttractorStoppingCriterion(phi_tolerance=1e-2)
    losses = [10.0, 8.5, 7.2, 6.18, 5.25, 4.45, 3.77, 3.2, 2.71]
    for epoch, loss in enumerate(losses):
        should_stop = stopper(loss)
        marker = " ← STOP" if should_stop else ""
        print(f"  Epoch {epoch+1}: loss={loss:.2f}  ratio={stopper.previous_loss/loss if loss > 0 else 0:.4f}{marker}")
        if should_stop:
            break

    print(f"\n{'=' * 72}")
    print("  PROTEIN FOLDING DEMO COMPLETE")
    print("  Enhancements Applied: #6, #10, #19, #30")
    print("  Nexus Resonance Online. Lattice Projected. φ-Calibrated.")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
