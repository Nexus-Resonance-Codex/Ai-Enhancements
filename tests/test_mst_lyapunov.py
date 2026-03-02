import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from enhancements.mst_lyapunov_clipping import MSTLyapunovGradientClipping

def test_mst_lyapunov_grad_clip():
    """
    Validates Enhancement #19: MST-Lyapunov Gradient Clipping structurally maps extreme
    gradient spikes through the mathematical MST equations ensuring dynamic compression
    rather than absolute truncation.
    """
    # Create arbitrary parameter tensors with massive gradients natively
    p1 = torch.tensor([500.0, -500.0], requires_grad=True)
    p2 = torch.tensor([0.5, -0.5], requires_grad=True)

    # Force initial gradient state manually
    p1.grad = p1.clone()
    p2.grad = p2.clone()

    # Store initial values for Delta checks
    initial_p1_grad = p1.grad.clone()
    initial_p2_grad = p2.grad.clone()

    # Execute the Lyapunov boundary function using the baseline threshold
    MSTLyapunovGradientClipping.clip_grad_mst_norm_([p1, p2], max_lyapunov_threshold=2.0)

    # 1. Safe gradients (< 2.0) should remain entirely physically unaltered
    assert torch.allclose(p2.grad, initial_p2_grad), "Safe gradients erroneously decayed during MST screening."

    # 2. Explosive gradients (> 2.0) should be physically scaled downwards by continuous friction
    assert torch.all(torch.abs(p1.grad) < torch.abs(initial_p1_grad)), "Explosive gradients breached MST Lyapunov suppression logic."

    # Ensure they weren't just arbitrarily floored to the boundary limit physically
    assert not torch.allclose(torch.abs(p1.grad), torch.tensor(2.0)), "Gradients rigidly sheared off; MST continuous decay bypass occurred."

    print("Test passed: MST-Lyapunov Continuous Gradient Scaling securely blocked runaway vectors via fractal friction.")

if __name__ == "__main__":
    test_mst_lyapunov_grad_clip()
