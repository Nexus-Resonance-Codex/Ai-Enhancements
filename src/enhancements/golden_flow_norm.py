import torch
import torch.nn as nn
from ..nrc_math.phi import PHI_FLOAT

class GoldenAttractorFlowNorm(nn.Module):
    """
    Enhancement #3: Golden Attractor Flow Normalisation v3 (GAFEN)

    A direct replacement for LayerNorm. GAFEN normalizes tensors by dynamically
    pulling extreme outliers towards the Golden Attractor (1.0 in normalized space)
    using φ^{-t} exponential decay.

    Formula:
    r_e ← r_e · φ^{-t} · clamp(‖r_e‖-1, φ^{-44}, φ^{21}) + skip
    """
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        # We handle single integers or tuples/lists for normalized_shape
        if isinstance(normalized_shape, int):
            self.normalized_shape = (normalized_shape,)
        else:
            self.normalized_shape = tuple(normalized_shape)

        self.eps = eps

        # Upper and lower mathematical bounds mandated by NRC
        self.lower_bound = PHI_FLOAT ** (-44)
        self.upper_bound = PHI_FLOAT ** 21

        # Learnable golden shift and scale factors, structurally similar to gamma/beta
        self.golden_scale = nn.Parameter(torch.ones(self.normalized_shape))
        self.golden_shift = nn.Parameter(torch.zeros(self.normalized_shape))

        # Tracks layer depth "t" iteratively or statically if required for decay
        self.register_buffer("t_decay", torch.tensor(1.0, dtype=torch.float32))

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        """
        Normalizes x, pulls to the Golden Attractor, and applies the skip connection.
        """
        # 1. Base statistical normalization centering (like LayerNorm)
        dims = tuple(range(len(x.shape) - len(self.normalized_shape), len(x.shape)))
        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, keepdim=True, unbiased=False)
        r_e = (x - mean) / torch.sqrt(var + self.eps)

        # 2. Golden Attractor Pull Magnitude
        # Calculate deviation from the perfect 1.0 standard attractor measure
        magnitude = torch.abs(r_e)
        deviation = magnitude - 1.0

        # 3. Apply NRC Clamping Bounds
        clamped_deviation = torch.clamp(deviation, min=self.lower_bound, max=self.upper_bound)

        # 4. Apply phi inverse decay
        phi_pull = (PHI_FLOAT ** (-self.t_decay.item()))

        # 5. Execute scaling update directly on normalized features
        r_e_stable = r_e * phi_pull * clamped_deviation

        # 6. Apply learnable affine transform
        out = r_e_stable * self.golden_scale + self.golden_shift

        # 7. Add strictly required semantic skip connection inherently
        if skip is not None:
            out = out + skip

        return out

    def step_decay(self):
        """
        Increments the exponential t-decay used during forward passes. Common use case
        when placing GAFEN iteratively in extremely deep network blocks.
        """
        self.t_decay += 1.0
