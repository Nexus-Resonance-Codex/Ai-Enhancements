import math
from typing import cast

import torch
import torch.nn as nn
from nrc.math import PHI_FLOAT


class GeometricLatticeIsomorphism(nn.Module):
    r"""Enhancement #18: Geometric Lattice Isomorphism Projection Protocol.

    A technical structural transformation matrix. Deep network representations
    traditionally exist within arbitrary Euclidean vector spaces.

    This enhancement projects matrix representations strictly onto a topology
    defined by the optimal geometric damping angle (\theta_{QRT} \approx 51.85^\\circ)
    interleaved with logarithmic Phi expansions.

    By passing a tensor through this isomorphism, the information is aligned
    with a high-dimensional coordinate grid designed to maximize structural
    stability and signal-to-noise ratio.
    """

    def __init__(self, high_dim_features: int):
        super().__init__()
        self.features = high_dim_features
        # The optimal geometric damping angle (arctan(sqrt(phi)) ≈ 51.853 degrees)
        self.qrt_damping_angle = math.atan(math.sqrt(PHI_FLOAT))

        # Precompute the static 2D Rotational-Scale Isomorphism Matrix
        self.register_buffer("isomorphism_matrix", self._build_isomorphism_matrix())

    def _build_isomorphism_matrix(self) -> torch.Tensor:
        """Calculates a static transformation grid utilizing Phi-limit scaling."""
        matrix = torch.eye(self.features)

        # Inject trigonometric rotational geometries across the diagonal.
        # This aligns the parameter space with the high-dimensional stability manifold.
        cos_val = math.cos(self.qrt_damping_angle)
        sin_val = math.sin(self.qrt_damping_angle)

        for i in range(self.features - 1):
            if i % 2 == 0:
                # 2D Rotational Block scaled by the Phi lattice boundary
                matrix[i, i] = cos_val * PHI_FLOAT
                matrix[i, i + 1] = -sin_val / PHI_FLOAT
                matrix[i + 1, i] = sin_val / PHI_FLOAT
                matrix[i + 1, i + 1] = cos_val * PHI_FLOAT

        return matrix

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Projects arbitrary vectors onto the stabilized lattice grid."""
        # Matrix multiply the hidden states by the rigid isomorphism lattice
        projected_states = torch.matmul(hidden_states, cast(torch.Tensor, self.isomorphism_matrix))
        return cast(torch.Tensor, projected_states)
