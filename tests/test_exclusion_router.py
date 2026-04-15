import torch
from nrc_math import PHI_FLOAT

from nrc_ai.exclusion_gradient_router import BiologicalExclusionGradientRouter


def test_exclusion_gradient_router() -> None:
    """Validates Enhancement #6: Exclusion Router logic.

    Ensures gradients are structurally masked by Mod-9 biological exclusion.
    """
    dim = 256
    router = BiologicalExclusionGradientRouter()

    # 1. Forward Pass
    # Create a leaf tensor to ensure .grad is populated
    # Create integer-aligned inputs to ensure we hit Mod 9 zones (0, 3, 6)
    # Use randint then cast to float for the nn.Module
    scaled_x = torch.randint(0, 100, (2, 16, dim)).float()
    scaled_x.requires_grad_(True)
    x = scaled_x  # Track for grad check
    output = router(scaled_x)

    assert output.shape == x.shape
    assert not torch.isnan(output).any()

    # 2. Path Integrity Check
    # Verify some indices are exactly 0.0 (structural exclusion)
    zero_count = (output == 0.0).sum().item()
    assert zero_count > 0, f"Exclusion gate failed to zero out chaotic paths. Zero count: {zero_count}"

    # 3. Backward Pass
    loss = output.sum()
    loss.backward()

    assert x.grad is not None
    # Survivors should have gradient = 1.0 * phi * 5000.0 (from input scaling)
    survivor_mask = output != 0.0
    grad_active = x.grad[survivor_mask]

    # Mathematical expectation: unit_grad * phi
    expected_grad = PHI_FLOAT
    assert torch.allclose(grad_active, torch.full_like(grad_active, expected_grad), rtol=1e-5), (
        f"Gradients were not correctly amplified! Found {grad_active[0].item()}, Expected {expected_grad}"
    )
