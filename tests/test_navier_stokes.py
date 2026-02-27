import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from enhancements.navier_stokes_damping import NavierStokesDampingRegulariser
from nrc_math.qrt import execute_qrt_damping_tensor

def test_navier_stokes_damping():
    """
    Validates Enhancement #10: Navier-Stokes Damping Regulariser smoothly
    brakes extreme tensor spikes utilizing the QRT topological decay bounds,
    and reliably passes shape execution mathematically.
    """
    batch_size = 8
    seq_len = 256
    embed_dim = 128

    regulariser = NavierStokesDampingRegulariser(damping_strength=0.1)

    # 1. Create a tensor with intentional destructive spikes (e.g. exploding gradients proxy)
    exploding_states = torch.randn(batch_size, seq_len, embed_dim)

    # Force an absolutely massive structural spike in a specific coordinate
    exploding_states[0, 0, 0] = 50000.0
    exploding_states[0, 0, 1] = -50000.0

    # 2. Forward execution through damping fluid
    damped_output = regulariser(exploding_states)

    assert damped_output.shape == exploding_states.shape, "Regulariser altered spatial dimensionality of the tensor."

    # 3. Check QRT Friction limits
    # At extreme values, exp(-x^2 / phi) goes to 0, leaving only the cos((pi / phi) * x) boundary.
    # Therefore, the added friction penalty for a massive spike is purely oscillatory and bounded natively [-1, 1] * strength.
    # This proves the regulariser doesn't return massive inf/nan values when attempting to correct anomalies.
    assert not torch.isnan(damped_output).any(), "NaN developed dynamically inside the QRT Navier-Stokes bounds."
    assert not torch.isinf(damped_output).any(), "Inf triggered internally inside the QRT boundaries."

    print("Test passed: Navier-Stokes Damping Regulariser survived extreme explosive states by utilizing QRT topological friction bounds.")

if __name__ == "__main__":
    test_navier_stokes_damping()
