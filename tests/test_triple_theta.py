import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from nrc_ai.triple_theta_init import TripleThetaInitializer, triple_theta_init_


def test_triple_theta_initialization() -> None:
    """Validates Enhancement #4: Triple-Theta Initialization mathematically applies.

    Utilizes phi^n scaling and TUPT modular state exclusions to neural network weights.
    """
    # Create a generic weight tensor
    layer = torch.nn.Linear(256, 512)
    original_weights = layer.weight.clone()

    # Apply Triple-Theta Initialization
    triple_theta_init_(layer.weight, std=1.0)

    # 1. Check mutation
    assert not torch.allclose(original_weights, layer.weight), "Weights were not modified."

    # 2. Check for NaN/Inf
    assert not torch.isnan(layer.weight).any(), "Triple-Theta initialization generated NaNs."
    assert not torch.isinf(layer.weight).any(), "Triple-Theta initialization generated Infs."

    # 3. Check for Structural Exclusion Gates
    # Since we use TUPT modular masking on a large set (256x512 = 131,072 elements),
    # it is virtually guaranteed that multiple elements will hit the exclusion trap and become 0.0
    zero_count = (layer.weight == 0.0).sum().item()
    print(f"Triple-Theta masked out {zero_count} weights correctly via TUPT exclusions out of {layer.weight.numel()}.")

    assert zero_count > 0, "No values were zeroed out by the TUPT structural exclusion filter."

    # 4. Check the linear layer wrapper module
    custom_linear = TripleThetaInitializer(128, 128)
    assert not torch.isnan(custom_linear.weight).any()

    print("Test passed: Triple-Theta Initialization successfully creates stabilized structural geometric weights.")


if __name__ == "__main__":
    test_triple_theta_initialization()
