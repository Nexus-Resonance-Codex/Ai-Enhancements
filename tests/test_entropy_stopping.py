import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from enhancements.entropy_stopping import EntropyAttractorStoppingCriterion

def test_entropy_stopping_criterion():
    """
    Validates Enhancement #30: The Early Stopping mechanism aborts training
    structurally ONLY when the loss delta maps physically onto the Phi or 1/Phi bounds.
    """
    # Initialize the theoretical evaluation criterion
    # We use a wide tolerance just to prove the physics locally
    criterion = EntropyAttractorStoppingCriterion(phi_tolerance=0.01)

    # Epoch 1: Massive arbitrary initial loss
    epoch_1_loss = 10.0
    assert not criterion(epoch_1_loss), "Criterion prematurely aborted at epoch 1."

    # Epoch 2: Standard non-resonant descent mapping.
    # 10.0 -> 8.0 (Ratio is 1.25. Not matching 1.618 or 0.618 limits)
    epoch_2_loss = 8.0
    assert not criterion(epoch_2_loss), "Criterion incorrectly executed stopping sequence outside of Phi resonance boundaries."

    # Epoch 3: The model stumbles upon the absolute mathematical Resonance Attractor.
    # We orchestrate the exact loss that yields fundamentally 0.618 geometric ratio.
    # 8.0 * (1 / 1.6180339) ~= 4.944

    epoch_3_loss = 8.0 * ((math.sqrt(5.0) - 1.0) / 2.0)

    # This step should trigger the True physical termination response.
    terminal_state = criterion(epoch_3_loss)

    assert terminal_state, "NRC Entropy criterion failed to identify the explicit 1/Phi resonance attractor collapse."

    print("Test passed: Entropy Early Stopping identified and halted training physics dynamically purely upon reaching continuous Phi-limit Attractor Boundaries.")

import math

if __name__ == "__main__":
    test_entropy_stopping_criterion()
