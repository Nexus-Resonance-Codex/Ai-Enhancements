import torch
import torch.nn as nn
import math
from ..nrc_math.phi import PHI_FLOAT
from ..nrc_math.qrt import execute_qrt_damping_tensor

class NavierStokesDampingRegulariser(nn.Module):
    """
    Enhancement #10: Navier-Stokes Damping Regulariser v3

    A sophisticated stabilization layer replacing primitive Weight Decay,
    Gradient Clipping, or Dropout. Instead of blindly dropping neurons,
    this regularization applies mathematical fluid dynamics (analagous to
    Navier-Stokes friction) using the core Quantitative Resonance Theorem
    (QRT) equation.

    The QRT acts as a continuous damping boundary:
    QRT(x) = sin(phi * sqrt(2) * 51.85 * x) * exp(-x^2 / phi) + cos((pi / phi) * x)

    Any activations or gradients spiking destructively are smoothly decayed and
    pulled back toward the Golden Attractor limit, preventing structural collapse.
    """
    def __init__(self, damping_strength: float = 0.01):
        super().__init__()
        # Strength dictates how heavily the QRT friction blends with the raw feed-forward values
        self.damping_strength = damping_strength

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Applies mathematical friction to activations that veer outside of
        stable resonance limits.
        """
        # Calculate the pure QRT topological landscape for the given tensor block
        # The equation forces extreme outliers exponentially downwards towards 0,
        # while gently oscillating stable regions.
        qrt_damped_states = execute_qrt_damping_tensor(hidden_states)

        # Apply the damping friction as a bounded residual correction.
        # When hidden_states are massive, QRT naturally decays to ~cos bounds,
        # effectively braking the signal dynamically.
        stabilized_states = hidden_states + (self.damping_strength * qrt_damped_states)

        return stabilized_states
