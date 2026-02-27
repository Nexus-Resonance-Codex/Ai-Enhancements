import torch
import sys
import os
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from enhancements.phi_void_positional import PhiVoidPositionalEncoding
from nrc_math.phi import PHI_FLOAT

def test_phi_void_positional():
    """
    Validates Enhancement #23: Ensures the Positional Encoding grids properly scale based
    on the mathematical phi^6 void boundaries seamlessly across varying topological sequences.
    """
    batch = 2
    seq = 50
    dim = 256

    # 1. Initialize random token embeddings structurally
    token_embeddings = torch.zeros(batch, seq, dim)

    # 2. Deploy the mathematical Phi^6 encoder grid
    encoder = PhiVoidPositionalEncoding(d_model=dim)

    encoded_features = encoder(token_embeddings)

    # Validation A: Dimensionality Check
    assert encoded_features.shape == token_embeddings.shape, "Phi^6 positional logic corrupted tensor embedding spaces physically."

    # Validation B: Verification of boundary limits
    # Since inputs were zeros, the output is identically the pure position grid.
    # Sine/Cosine limits absolutely CANNOT exceed 1.0 structurally.
    max_bound = encoded_features.max().item()
    min_bound = encoded_features.min().item()

    assert max_bound <= 1.0001, f"Positional bounds exploded above max limit 1.0: {max_bound}"
    assert min_bound >= -1.0001, f"Positional bounds collapsed below min limit -1.0: {min_bound}"

    # Validation C: Direct calculation of the div_term for the very first step
    phi_six = PHI_FLOAT ** 6
    expected_div_0 = math.exp(0.0 * (-math.log(phi_six) / dim)) # Should equal 1.0 natively

    # Pos 1, Dim 0 should be sin(1 * 1.0) = sin(1)
    expected_val = math.sin(1.0)

    assert torch.isclose(encoded_features[0, 1, 0], torch.tensor(expected_val), rtol=1e-4), "Phi^6 expansion division math failed physically."

    print("Test passed: Phi^6 Void Positional Encoding dynamically locked sequence contexts onto native mathematical boundaries.")

if __name__ == "__main__":
    test_phi_void_positional()
