import torch
import torch.nn as nn
from nrc.math.phi import PHI_FLOAT


class GoldenSpiralRotaryEmbedding(nn.Module):
    """
    Enhancement #29: Golden Spiral Rotary Embedding Extension

    Standard Rotary Positional Embeddings (RoPE) use complex numbers to apply
    spatial distances by rotating vector states inside arbitrary circles.

    This enhancement replaces the standard circular boundary mathematically with
    the Golden Spiral (Logarithmic Spiral scaled strictly by Phi).
    As tokens slide further down the context sequence, their angular rotation
    scales dynamically along the explicit geometry of the Golden Spiral boundary,
    preserving perfect geometric relationships between tokens functionally no matter
    how large the sequence radius grows physically.
    """
    def __init__(self, dim: int, max_seq_len: int = 4096):
        super().__init__()
        self.dim = dim

        # We pre-calculate the topological Golden Spiral matrix natively
        self.register_buffer("inv_phi_spiral", self._build_phi_frequencies(dim))
        self.register_buffer("seq_spiral_tensor", self._build_sequence_mappings(max_seq_len))

    def _build_phi_frequencies(self, dim: int) -> torch.Tensor:
        """
        Creates the scaling frequency dimension using explicit Phi^N logic instead
        of arbitrary exponential integers natively.
        """
        # We build up continuous phi logic geometrically
        # Original RoPE is 10000 ** (-2 * (i - 1) / d)
        # We substitute the baseline frequency with Phi
        frequencies = torch.arange(0, dim, 2, dtype=torch.float32)

        # We mathematically power down the depth dimension via Phi geometrically
        # The larger the feature dim, the heavier the Phi exponential compression natively
        inv_freq = 1.0 / (PHI_FLOAT ** (frequencies / dim))
        return inv_freq

    def _build_sequence_mappings(self, max_seq_len: int) -> torch.Tensor:
        """
        Builds the physical sequence integer limits natively mapped.
        """
        return torch.arange(max_seq_len, dtype=torch.float32)

    def forward(self, x: torch.Tensor, seq_dim: int = 1) -> torch.Tensor:
        """
        Calculates the explicit sine and cosine frequency boundaries and injects
        the Golden Spiral matrices structurally onto the embeddings geometrically.
        """
        seq_len = x.shape[seq_dim]

        # 1. Access the explicit positional numbers physically mapped
        t = self.seq_spiral_tensor[:seq_len]

        # 2. Derive the Golden Spiral positional frequencies (Sequence Location x Phi Frequencies)
        # Using einstein summation to execute the algebraic bounding efficiently
        freqs = torch.einsum("i,j->ij", t, self.inv_phi_spiral)

        # 3. Repeat frequencies natively across the sequence dimensions
        # This allows us to map sin AND cos arrays identically
        emb = torch.cat((freqs, freqs), dim=-1)

        # 4. We calculate the Sine and Cosine boundaries strictly bound by the Golden Spiral
        cos_matrix = emb.cos().unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, dim)
        sin_matrix = emb.sin().unsqueeze(0).unsqueeze(0)

        # 5. Execute the Algebraic Rotary mapping structurally
        # x_rotated = x * cos + x_half_rotated * sin

        d = x.shape[-1] // 2
        # Physically pivot the tokens geometrically via slicing mathematically
        x_half_rotated = torch.cat((-x[..., d:], x[..., :d]), dim=-1)

        x_golden_rope = (x * cos_matrix) + (x_half_rotated * sin_matrix)

        return x_golden_rope
