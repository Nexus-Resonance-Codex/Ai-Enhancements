import math

import torch
import torch.nn as nn


class QRTSoftmax(nn.Module):
    """Quantum Resonance Theory Softmax.

    Instead of standard softmax which pushes probability purely based on exponential
    magnitudes, QRTSoftmax applies the QRT wave function multiplier to logits before
    normalization, pulling logits toward the golden limit and clamping the
    chaotic void nodes (0, 3, 6, 9).
    """

    def __init__(self, phi=1.6180339887):
        super().__init__()
        self.phi = phi
        self.phi_inv = 1.0 / phi

    def forward(self, logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
        # Applying the QRT Dampening to logits directly
        # QRT(x) = sin(phi * sqrt(2) * 51.85 * x) * exp(-x^2 / phi) + cos(pi / phi * x)

        # Simplified QRT tensor expansion for efficiency
        x = logits
        # Limit exponential blowup
        x_clamped = torch.clamp(x, -20.0, 20.0)

        # QRT Wave multiplier
        freq = self.phi * math.sqrt(2) * 51.85
        wave = torch.sin(freq * x_clamped) * torch.exp(-(x_clamped**2) * self.phi_inv) + torch.cos(
            (math.pi * self.phi_inv) * x_clamped
        )

        # The QRT wave naturally amplifies resonance and dampens dissonance
        # Modulate the original logits
        qrt_logits = logits * (1.0 + 0.1 * wave)

        # Enforce 0-3-6-9 chaotic void avoidance on the last dimension
        vocab_size = qrt_logits.size(dim)
        indices = torch.arange(vocab_size, device=logits.device)
        mod_9 = indices % 9

        # Apply heavy penalty to chaotic indices
        mask = torch.ones_like(indices, dtype=logits.dtype)
        mask[(mod_9 == 0) | (mod_9 == 3) | (mod_9 == 6)] = -1e9  # 9 % 9 is 0

        # Broadcast mask correctly if dim is -1 and logits has multiple dims
        if dim == -1 or dim == logits.dim() - 1:
            qrt_logits = qrt_logits + mask

        return torch.nn.functional.softmax(qrt_logits, dim=dim)
