import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from enhancements.gtt_entropy_regulariser import GTTEntropyCollapseRegulariser
from nrc_math.phi import PHI_FLOAT

def test_gtt_entropy_collapse():
    """
    Validates Enhancement #12: The GTT Entropy Collapse layer successfully
    measures real-time Shannon Entropy and proportionally scales down specific
    vectors that exceed the 10.96 nat boundary limits.
    """
    batch_size = 2
    seq_len = 4
    embed_dim = 65000  # Require a massive dimensionality to realistically hit 10.96 Shannon nats

    regulariser = GTTEntropyCollapseRegulariser(gtt_safe_boundary=10.96)

    # 1. Create a perfectly uniform massive tensor.
    # A completely uniform distribution of size N has maximum entropy: H = ln(N).
    # ln(65000) = 11.08 nats. This WILL EXCEED the 10.96 boundary natively.
    high_entropy_uniform_states = torch.zeros(batch_size, seq_len, embed_dim)

    # Calculate pure entropy manually to verify logic:
    # probs = 1.0 / 65000
    # H = - sum(65000 * (1/65000 * ln(1/65000))) = ln(65000) ~ 11.082

    # 2. Create another batch with extremely peaked (low entropy) data.
    low_entropy_peaked_states = torch.zeros(batch_size, seq_len, embed_dim)
    low_entropy_peaked_states[:, :, 0] = 50.0  # Force a massive spike so Softmax -> 1.0 for index 0. Entropy ~ 0.

    # 3. Test High Entropy (Should collapse by 1/Phi)
    # The uniform states originally had values of 0.0.
    # Actually wait - to see the scaling effectively, we shouldn't use 0.0 (since 0.0 / phi = 0.0).
    # Let's populate the tensor with a uniform constant instead:
    high_entropy_uniform_states = torch.ones(batch_size, seq_len, embed_dim) * 2.5

    damped_output = regulariser(high_entropy_uniform_states)

    # Verify the value collapsed by phi
    # Since all tokens breached the 10.96 boundary, all values should scale.
    expected_damped_val = 2.5 / PHI_FLOAT
    assert torch.allclose(damped_output, torch.full_like(damped_output, expected_damped_val)), "High-entropy tensor blocked the phi-collapse regularisation scaling."

    # 4. Test Low Entropy (Should remain unaltered)
    clean_output = regulariser(low_entropy_peaked_states)
    assert torch.allclose(clean_output, low_entropy_peaked_states), "Low-entropy tensor illegally collapsed during pass-through."

    print("Test passed: GTT Entropy monitor dynamically sensed thermodynamic 10.96 breaches and correctly deployed Phi Shard compression.")

if __name__ == "__main__":
    test_gtt_entropy_collapse()
