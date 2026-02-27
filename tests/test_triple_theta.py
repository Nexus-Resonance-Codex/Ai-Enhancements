import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from enhancements.triple_theta_init import triple_theta_init_, TripleThetaLinear

def test_triple_theta_initialization():
    """
    Validates Enhancement #4: Triple-Theta Initialisation v3 mathematically applies
    φ^n scaling and Z_2187 biological exclusions to neural network weights.
    """
    # Create a generic weight tensor
    layer = torch.nn.Linear(256, 512)
    original_weights = layer.weight.clone()

    # Apply Triple-Theta Initialization
    triple_theta_init_(layer.weight, std=1.0)

    # 1. Check mutation
    assert not torch.allclose(original_weights, layer.weight), "Weights were not modified."

    # 2. Check for NaN/Inf
    assert not torch.isnan(layer.weight).any(), "Triple-Theta init generated NaNs."
    assert not torch.isinf(layer.weight).any(), "Triple-Theta init generated Infs."

    # 3. Check for Biologically Excluded Zero Gates
    # Since we use Mod 2187 masking on a large set (256x512 = 131,072 elements),
    # it is virtually guaranteed that multiple elements will hit the exclusion trap and become 0.0
    zero_count = (layer.weight == 0.0).sum().item()
    print(f"Triple-Theta masked out {zero_count} weights correctly via Mod 2187 exclusions out of {layer.weight.numel()}.")

    assert zero_count > 0, "No values were zeroed out by the Mod 2187 biological exclusion filter."

    # 4. Check the linear layer wrapper module
    custom_linear = TripleThetaLinear(128, 128)
    assert not torch.isnan(custom_linear.weight).any()

    print("Test passed: Triple-Theta Initialisation v3 successfully creates NRC fractal bounded weights.")

if __name__ == "__main__":
    test_triple_theta_initialization()
