import torch

from nrc_ai.giza_isomorphism import GizaLatticeIsomorphism


def test_giza_isomorphism() -> None:
    """Validates Giza-Lattice Isomorphism diagonal identity (cos * phi = 1.0)."""
    dim = 128
    layer = GizaLatticeIsomorphism(high_dim_features=dim)

    # 1. Identity Projection
    x = torch.zeros(1, dim)
    x[0, 0] = 1.0
    out = layer(x)

    # Diagonal[0,0] = cos(theta) * phi = (1/phi) * phi = 1.0
    assert torch.isclose(out[0, 0], torch.tensor(1.0), rtol=1e-5), (
        f"Giza Isomorphism scale breach. Found: {out[0, 0].item()}"
    )
