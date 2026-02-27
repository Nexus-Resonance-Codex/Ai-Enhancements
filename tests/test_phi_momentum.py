import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from enhancements.phi_momentum_accelerator import PhiInverseMomentumAccelerator
from nrc_math.phi import PHI_FLOAT

def test_phi_momentum():
    """
    Validates Enhancement #13: Checks if gradient updates successfully scale
    upwards by phi on continuous resonant tracks, and decay downwards by
    1/phi upon detecting oscillation barriers.
    """
    # Create two parameter tensors simulating weights
    p_increasing = torch.tensor([1.0], requires_grad=True)
    p_oscillating = torch.tensor([1.0], requires_grad=True)

    optimizer = PhiInverseMomentumAccelerator([p_increasing, p_oscillating], lr=0.1, beta=0.0)

    # Force step 1 (Initialize momentum in positive direction)
    p_increasing.grad = torch.tensor([1.0])
    p_oscillating.grad = torch.tensor([1.0])

    optimizer.step()

    # Zero gradients
    optimizer.zero_grad()

    # Store intermediate state
    intermediate_increasing = p_increasing.clone().detach()
    intermediate_oscillating = p_oscillating.clone().detach()

    # Step 2:
    # Tensor 1 continues in the positive direction (Resonant Acceleration)
    p_increasing.grad = torch.tensor([1.0])
    # Tensor 2 violently swings negative (Oscillating Dampen)
    p_oscillating.grad = torch.tensor([-1.0])

    optimizer.step()

    # Validate the positional updates
    delta_increasing = torch.abs(intermediate_increasing - p_increasing)
    delta_oscillating = torch.abs(intermediate_oscillating - p_oscillating)

    # The absolute structural update of the accelerating tensor should be phi^2 times larger
    # than the damped oscillating tensor due to the (phi) vs (1/phi) routing mechanism.
    expected_ratio = PHI_FLOAT ** 2
    actual_ratio = (delta_increasing / delta_oscillating).item()

    print(f"Momentum tracking expected a {expected_ratio:.4f}x acceleration margin across the topological boundaries.")
    assert torch.isclose(torch.tensor(actual_ratio), torch.tensor(expected_ratio), rtol=1e-3), "Phi scaling router failed in Momentum tracking."
    print("Test passed: Phi-Inverse Momentum dynamically shaped backpropagation bounds via the Golden Ratio.")

if __name__ == "__main__":
    test_phi_momentum()
