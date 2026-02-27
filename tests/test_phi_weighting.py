import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from enhancements.phi_resonant_weighting import PhiResonantWeighting
from nrc_math.phi import PHI_FLOAT

def test_phi_resonant_weighting():
    """
    Validates Enhancement #17: Ensures that the Phi Resonant Weighting logic
    correctly senses low variance (scaling by phi) and high variance (damping by 1/phi).
    """
    features = 128
    batch_size = 2
    seq_len = 10

    layer = PhiResonantWeighting(in_features=features)

    # Simulate a collapsing tensor (near-zero variance)
    collapsed_states = torch.ones(batch_size, seq_len, features) * 0.5
    # Add tiny noise so variance isn't literally 0.0 but remains strictly < 1.0
    collapsed_states += torch.randn_like(collapsed_states) * 0.1

    # Simulate an exploding tensor
    exploding_states = torch.randn(batch_size, seq_len, features) * 100.0

    # Track the pure mathematical scaling before the learnable gate adapts
    damped_output = layer(exploding_states)
    accelerated_output = layer(collapsed_states)

    assert not torch.isnan(damped_output).any(), "Resonant damping generated NaNs"
    assert not torch.isnan(accelerated_output).any(), "Resonant acceleration generated NaNs"

    # The output shapes should be completely intact
    assert damped_output.shape == exploding_states.shape, "Layer mutated spatial tracking logic"

    # Since the initial learnable gate is exactly 1.0, and the variance boundary logic routed cleanly:
    # 1. Collapsed states should have multiplied identically by phi.
    assert torch.allclose(accelerated_output, collapsed_states * PHI_FLOAT, rtol=1e-4), "Collapsing layer did not cleanly multiply by phi"

    # 2. Exploding states should have multiplied identically by 1/phi.
    assert torch.allclose(damped_output, exploding_states * (1.0 / PHI_FLOAT), rtol=1e-4), "Exploding layer did not cleanly damp by 1/phi"

    print("Test passed: Phi Resonant Weighting dynamically monitored structural variance and scaled via Golden bounds.")

if __name__ == "__main__":
    test_phi_resonant_weighting()
