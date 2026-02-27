import torch
import torch.nn as nn
from ..nrc_math.phi import PHI_FLOAT

class PhiResonantWeighting(nn.Module):
    """
    Enhancement #17: phi-Powered Resonant Weighting

    A structural layer wrapper for applying adaptive fractal resonance.
    Standard Neural Networks apply equal learning agency across all layers, leading
    to catastrophic forgetting or uncoordinated topology.

    This enhancement forces intermediate block activations to dynamically scale by
    phi (Golden Ratio) or its inverse (1/phi) depending on the sequence dimensionality.
    By doing so, tensors explicitly align their internal variance with the mathematical
    attractor fields of the NRC, establishing true topological harmony across extreme sizes.
    """
    def __init__(self, in_features: int):
        super().__init__()
        # A learnable parameter bounding the extent of the Phi shift
        self.resonance_gate = nn.Parameter(torch.ones(in_features))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch_size, seq_len, in_features)
        """
        # We calculate the normalized variance of the current block
        variance = torch.var(hidden_states, dim=-1, keepdim=True)

        # If the variance collapses below critical density, we accelerate it outward structurally using Phi.
        # If the variance explodes unhealthily, we damp it proportionally using 1/Phi.
        # This replaces traditional static LayerNorm scaling with a dynamic fractal attractor.
        scale_factor = torch.where(variance < 1.0, PHI_FLOAT, 1.0 / PHI_FLOAT)

        # Merge the learned gating mechanism with the exact logical math limit
        adaptive_weighting = self.resonance_gate * scale_factor

        resonant_states = hidden_states * adaptive_weighting

        return resonant_states
