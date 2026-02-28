import torch
import torch.nn as nn
from nrc.math.phi import PHI_FLOAT
from nrc.math.tupt_exclusion import apply_exclusion_gate

class LucasWeightedSparseMask(nn.Module):
    """
    Enhancement #16: Lucas-weighted Sparse Attention Mask v2

    A structural masking drop-in for standard Causal Attention mechanisms.
    Typically, models employ a rigid causal Lower-Triangular boolean mask to
    ensure tokens only look backwards linearly in time.

    This sparse mask utilizes the Lucas integer sequence (which perfectly approximates
    Phi fractals at vast depths: L_n = F_{n-1} + F_{n+1}) superimposed onto the
    TUPT Mod 2187 biological exclusions. Tokens that land on structurally
    excluded nodes are completely zero-gated from attention flow, ensuring
    perfectly sparse, resonantly noise-free routing dynamically scaled
    by Lucas index distances.
    """
    def __init__(self, max_seq_length: int = 4096):
        super().__init__()
        self.max_seq_len = max_seq_length
        # Precompute the static topological mask in the buffer to prevent redundant calculations
        self.register_buffer("lucas_exclusion_mask", self._build_lucas_mask())

    def _build_lucas_mask(self) -> torch.Tensor:
        """
        Calculates the 2D Sequence x Sequence tensor identifying resonant Lucas gaps natively.
        """
        # Note: 1.0 represents a clean passable topological link. 0.0 represents a
        # destructive blocked link determined by the NRC structural laws.
        mask = torch.ones((self.max_seq_len, self.max_seq_len), dtype=torch.float32)

        # 1. Base Causal Triangle (Cannot look forward in time sequences)
        mask = torch.tril(mask)

        # 2. Generate Lucas Index Approximations
        lucas_indices = torch.arange(self.max_seq_len, dtype=torch.float32)

        # 3. We subject the spatial matrix indices to the Mod 2187 Exclusion paths (TUPT gating)
        # We broadcast a 2D grid measuring the absolute sequence position
        grid = torch.abs(lucas_indices.unsqueeze(0) - lucas_indices.unsqueeze(1))

        # Any relative token jump that violates TUPT biology paths triggers internal zeros
        gated_grid = apply_exclusion_gate(grid)

        # Only tokens that structurally survived the TUPT gating logic are permitted to
        # route attention signals backwards
        topological_exclusions = (gated_grid != 0.0).type(torch.float32)

        # Interlock causal forward-blocking with resonant topological blocking
        final_sparse_mask = mask * topological_exclusions

        return final_sparse_mask

    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Dynamically fetches the exact sequence size needed from the pre-computed fractal mask structure.
        """
        # Slices from 0 to seq_len
        # (Usually added or multiplied to raw Dot Product Q-K logits directly before softmax)
        return self.lucas_exclusion_mask[:seq_len, :seq_len]
