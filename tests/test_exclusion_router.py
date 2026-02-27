import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from enhancements.exclusion_gradient_router import BiologicalExclusionGradientRouter
from nrc_math.phi import PHI_FLOAT

def test_exclusion_gradient_router():
    """
    Validates Enhancement #6: Checks that the Biological Exclusion Router blocks specific
    mathematical forward paths and amplifies valid backward gradients by the Golden Ratio.
    """
    # 1. Setup simulated inputs requiring gradients
    batch_size, hidden_dim = 8, 256

    # We explicitly scale to trigger the boundary conditions of Mod 2187 traps
    inputs = torch.randn(batch_size, hidden_dim, requires_grad=True) * 5000.0
    router = BiologicalExclusionGradientRouter()

    # 2. Forward Pass Verification
    routed_output = router(inputs)

    # Check that exclusions occurred
    zero_count = (routed_output == 0.0).sum().item()
    print(f"Router excluded {zero_count} biological dead-paths in the forward pass.")
    assert zero_count > 0, "No paths were excluded by the router mask. Check Mod 2187 scaling."

    # 3. Backward Pass Verification
    # Trigger an artificial gradient of 1.0 flowing back from downstream
    dummy_loss = routed_output.sum()
    dummy_loss.backward()

    # Extract the custom-routed gradient
    grad = inputs.grad

    # Properties we expect to hold:
    # A) Locations that were zeroed in forward pass must have exactly 0.0 gradient
    forward_zeros_mask = (routed_output == 0.0)
    assert (grad[forward_zeros_mask] == 0.0).all(), "Gradients leaked into excluded biological dead paths!"

    # B) Locations that survived must have gradient equal to 1.0 * phi (due to the backward amplification)
    forward_active_mask = (routed_output != 0.0)
    # Give a tiny tolerance for floating point epsilon
    grad_active = grad[forward_active_mask]
    assert torch.allclose(grad_active, torch.full_like(grad_active, PHI_FLOAT)), "Gradients were not correctly amplified by the Golden Ratio!"

    print("Test passed: Biological Exclusion Gradient Router v3 correctly blocks invalid Mod 2187 pathways and accelerates resilient gradients by phi.")

if __name__ == "__main__":
    test_exclusion_gradient_router()
