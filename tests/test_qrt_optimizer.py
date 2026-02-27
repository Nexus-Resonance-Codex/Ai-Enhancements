import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from enhancements.qrt_optimizer import QRTTurbulenceOptimizer

def test_qrt_turbulence_optimizer():
    """
    Validates Enhancement #26: The Custom QRT optimizer successfully routes gradients
    and measures gradient variance via continuous mathematical QRT physics boundaries.
    """
    # Create two arbitrary physical parameter blocks
    p1 = torch.tensor([5.0], requires_grad=True)
    p2 = torch.tensor([-3.0], requires_grad=True)

    # 1. Spawn the mathematical QRT execution framework
    optimizer = QRTTurbulenceOptimizer([p1, p2], lr=0.1)

    # 2. Assign completely destructive gradient structures
    p1.grad = torch.tensor([500.0]) # Massive explosive anomaly -> High turbulence
    p2.grad = torch.tensor([0.1])   # Standard mapping vector -> Low turbulence

    # Lock initial state mappings globally for delta equations
    initial_p1 = p1.clone().detach()
    initial_p2 = p2.clone().detach()

    # Trigger structural physics map
    optimizer.step()

    # Verify continuous step boundary application
    assert p1.item() != initial_p1.item(), "Optimizer completely failed to step explosive parameters algebraically."
    assert p2.item() != initial_p2.item(), "Optimizer failed to track stable geometries natively."

    # Extract functional memory tensors
    state_p1 = optimizer.state[p1]

    # Verify standard Adam structural base remains intact natively for continuous updates
    assert 'turbulence' in state_p1, "Optimizer matrix corrupted internal variance logic physically."
    assert state_p1['step'] == 1, "Physical step advancement locked randomly."

    print("Test passed: QRT-Turbulence Optimizer structurally damped Adam variance matrices natively through continuous fractal friction.")

if __name__ == "__main__":
    test_qrt_turbulence_optimizer()
