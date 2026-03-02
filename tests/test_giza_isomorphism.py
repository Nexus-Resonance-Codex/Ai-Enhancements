import torch
import sys
import os
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from enhancements.giza_isomorphism import GizaLatticeIsomorphism
from nrc_math.phi import PHI_FLOAT

def test_giza_isomorphism():
    """
    Validates Enhancement #18: The Isomorphism Protocol mathematically rigidly projects
    Euclidean states onto the Giza-coordinate grid.
    """
    dim_size = 128
    batch_batch = 2

    layer = GizaLatticeIsomorphism(high_dim_features=dim_size)

    # Send a flat structural vector into the projection grid
    flat_states = torch.ones(batch_batch, dim_size)

    giza_projected = layer(flat_states)

    assert giza_projected.shape == flat_states.shape, "Dimensionality corrupted inside Isomorphism Matrix."

    # Mathematical Validation of the 2D Spin blocks
    # Index 0 and 1 are spun algebraically by:
    # y0 = (cos*phi) * x0 + (sin/phi) * x1
    # y1 = (-sin/phi) * x0 + (cos*phi) * x1

    cos_val = math.cos(51.85 * (math.pi / 180.0))
    sin_val = math.sin(51.85 * (math.pi / 180.0))

    expected_y0 = (cos_val * PHI_FLOAT) * 1.0 + (sin_val / PHI_FLOAT) * 1.0
    expected_y1 = (-sin_val / PHI_FLOAT) * 1.0 + (cos_val * PHI_FLOAT) * 1.0

    assert torch.isclose(giza_projected[0, 0], torch.tensor(expected_y0), rtol=1e-4), "Giza Projection failed rotational mathematics."
    assert torch.isclose(giza_projected[0, 1], torch.tensor(expected_y1), rtol=1e-4), "Giza Projection failed topological boundaries."

    print("Test passed: Giza-Lattice Isomorphism statically bounded generic tensors entirely onto the strict 51.85-degree Phi grid.")

if __name__ == "__main__":
    test_giza_isomorphism()
