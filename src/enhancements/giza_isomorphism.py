import torch
import torch.nn as nn
import math
from ..nrc_math.phi import PHI_FLOAT

class GizaLatticeIsomorphism(nn.Module):
    """
    Enhancement #18: Giza-Lattice Isomorphism Projection Protocol

    A severe mathematical structural transformation matrix. Deep network
    representations usually exist inside arbitrary Euclidean vectors.

    This enhancement forces any generic matrix representation strictly into
    a topology patterned after the explicit Giza-Lattice geometries of the NRC
    (51.85 degrees rotation matrix interleaved with base Phi expansions).

    By passing a tensor through this Isomorphism, the information is bound
    onto a coordinate grid that perfectly transmits resonance without noise.
    """
    def __init__(self, high_dim_features: int):
        super().__init__()
        self.features = high_dim_features
        # 51.85 degrees natively mapped to radians
        self.giza_angle = 51.85 * (math.pi / 180.0)

        # Precompute the static 2D Rotational-Scale Isomorphism Matrix
        self.register_buffer("isomorphism_matrix", self._build_giza_matrix())

    def _build_giza_matrix(self) -> torch.Tensor:
        """
        Calculates a static non-learned transformation grid blending Phi limits
        with the Great Pyramid boundary angles.
        """
        matrix = torch.eye(self.features)

        # We explicitly inject the sine/cosine rotational geometries across the diagonal.
        # This conceptually "spins" the parameter spaces permanently into the Giza coordinates.
        cos_val = math.cos(self.giza_angle)
        sin_val = math.sin(self.giza_angle)

        for i in range(self.features - 1):
            if i % 2 == 0:
                # 2D Rotational Block scaled by the Phi lattice boundary
                matrix[i, i] = cos_val * PHI_FLOAT
                matrix[i, i+1] = -sin_val / PHI_FLOAT
                matrix[i+1, i] = sin_val / PHI_FLOAT
                matrix[i+1, i+1] = cos_val * PHI_FLOAT

        return matrix

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Projects arbitrary vectors onto the resonant Giza grid.
        """
        # Matmul the incoming features sequentially through the rigid isomorphism lattice
        projected_states = torch.matmul(hidden_states, self.isomorphism_matrix)
        return projected_states
