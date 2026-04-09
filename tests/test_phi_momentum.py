import torch
from nrc_ai.phi_momentum_accelerator import PhiInverseMomentumAccelerator

def test_phi_momentum() -> None:
    """Validates Phi-Inverse Momentum stability and scaling."""
    param = torch.nn.Parameter(torch.ones(10, 10))
    optimizer = PhiInverseMomentumAccelerator([param], lr=0.01)
    
    # Simulate gradient update
    loss = (param**2).sum()
    loss.backward()
    
    initial_val = param.data.clone()
    optimizer.step()
    
    # Verify directional movement
    assert not torch.allclose(param.data, initial_val), "Momentum stall detected."
    assert not torch.isnan(param.data).any(), "Atomic explosion detected in momentum space."
