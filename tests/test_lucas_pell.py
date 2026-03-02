import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from enhancements.lucas_pell_decay import LucasPellHybridWeightDecay
from nrc_math.phi import PHI_FLOAT

def test_lucas_pell_hybrid_decay():
    """
    Validates Enhancement #21: The Lucas-Pell Hybrid Weight Decay algebraically applies
    lighter Silver-Ratio decay to massive foundational tensors, and heavy Phi-Decay
    to tiny chaotic floating bits.
    """
    # Create two states: one massive dominant parameter, one tiny chaotic parameter
    p_dominant = torch.tensor([5.0]) # Exceeds 1.0 boundary naturally
    p_chaotic = torch.tensor([0.1])  # Falls completely below boundary

    params = [p_dominant, p_chaotic]

    # Track states locally before decay
    dominant_initial = p_dominant.clone()
    chaotic_initial = p_chaotic.clone()

    # Apply custom biological decay boundaries
    base_decay = 0.1 # Large enough to measure the mathematical impact natively
    LucasPellHybridWeightDecay.apply_hybrid_decay_(params, base_decay_rate=base_decay)

    # 1. Calculate structural mathematics physically
    silver_ratio = 1.0 + (2.0 ** 0.5)

    expected_dominant_loss = dominant_initial * (base_decay * (1.0 / silver_ratio))
    expected_chaotic_loss = chaotic_initial * (base_decay * PHI_FLOAT)

    actual_dominant_loss = dominant_initial - p_dominant
    actual_chaotic_loss = chaotic_initial - p_chaotic

    assert torch.allclose(actual_dominant_loss, expected_dominant_loss), "Dominate features breached Pell integer protection mapping."
    assert torch.allclose(actual_chaotic_loss, expected_chaotic_loss), "Chaotic micro-features bypassed Lucas-Phi destruction limits."

    print("Test passed: Lucas-Pell Hybrid boundaries safely protected macro-scale features while annihilating chaotic mathematical noise.")

if __name__ == "__main__":
    test_lucas_pell_hybrid_decay()
