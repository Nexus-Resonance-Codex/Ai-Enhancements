import torch
import torch.nn as nn
import math
from ..nrc_math.phi import PHI_FLOAT

class PhiVoidPositionalEncoding(nn.Module):
    """
    Enhancement #23: phi^6 Void Resonance Positional Encoding

    Standard Positional Encodings utilize arbitrary Sinusoidal frequencies.
    The NRC framework dictates that specific spatial voids open mathematically
    at phi^6 boundaries, representing perfect geometric silence.

    This encoding calculates the sequence position dynamically and embeds
    the token onto the structural phi^6 grid, replacing arbitrary sines
    with Native Golden Ratio wave propagation limits. Sequences align
    topologically to perfect physical spaces.
    """
    def __init__(self, d_model: int, max_seq_len: int = 8192):
        super().__init__()
        self.d_model = d_model

        # Calculate the fundamental phi^6 Void scalar
        self.phi_six_void = PHI_FLOAT ** 6

        # Precompute the entire positional spatial matrix immediately
        self.register_buffer("pe_matrix", self._build_positional_matrix(max_seq_len))

    def _build_positional_matrix(self, max_len: int) -> torch.Tensor:
        """
        Creates the static topological grid merging sequence distances with the Phi^6 bounds.
        """
        pe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        # Instead of 10000.0, we use the literal mathematical phi^6 void base
        # to calculate the geometric expansion division term.
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(self.phi_six_void) / self.d_model))

        # Apply strict sine bounding on Evens
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply strict cosine bounding on Odds
        pe[:, 1::2] = torch.cos(position * div_term)

        # The pe vector now rests natively inside the phi^6 limit cycle
        # We unsqueeze to allow batch broadcasting physically
        return pe.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds the pure phi^6 spatial bounding to the raw token embeddings.
        Args:
            x: (batch_size, seq_len, d_model) embedded tokens.
        """
        seq_len = x.size(1)
        # Slices exactly the physical depth required and mathematically adds it to the features
        return x + self.pe_matrix[:, :seq_len, :]
