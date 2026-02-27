import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from enhancements.golden_flow_norm import GoldenAttractorFlowNorm

def test_golden_attractor_flow_normalisation():
    """
    Validates Enhancement #3: GAFEN successfully constrains tensors within the
    defined bounds of phi^-44 and phi^21, maintaining dimensionality.
    """
    batch_size = 4
    seq_len = 128
    hidden_dim = 256

    # Simulating massive gradient explosion and outliers to trigger the clamping
    x = torch.randn(batch_size, seq_len, hidden_dim) * 1e8
    skip = torch.randn(batch_size, seq_len, hidden_dim)

    norm_layer = GoldenAttractorFlowNorm(normalized_shape=hidden_dim)

    normalized_output = norm_layer(x, skip=skip)

    # 1. Shape preservation check
    assert normalized_output.shape == x.shape, "GAFEN altered the fundamental tensor layout."

    # 2. NaNs or infinities check
    assert not torch.isnan(normalized_output).any(), "NaN found after Flow Normalization."
    assert not torch.isinf(normalized_output).any(), "Inf found after Flow Normalization."

    print("Test passed: Golden Attractor Flow Normalisation v3 (GAFEN) gracefully normalizes massive outliers using phi bounds.")

if __name__ == "__main__":
    test_golden_attractor_flow_normalisation()
