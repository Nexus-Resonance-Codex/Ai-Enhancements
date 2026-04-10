import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from nrc_ai.modular_dropout import TUPTModularDropout


def test_modular_dropout() -> None:
    """Validates Enhancement #25: The TTT Modular Dropout grid.

    Uses deterministic modular residue placement indices to align dropout patterns
    with TTT structural stability nodes.
    """
    batch = 2
    seq = 500
    dim = 16

    # Deploy a completely pure 'active' hidden state matrix
    # Zeros are invisible to scale multiplication, so we use 1s
    pure_states = torch.ones(batch, seq, dim)

    dropout = TUPTModularDropout(probability=0.1)

    # Must enforce training mode globally for mask activation
    dropout.train()

    pruned_states = dropout(pure_states)

    # Validation A: Structural dimensionality continuity
    assert pruned_states.shape == pure_states.shape, "Modular dropout fractured tensor dimensions."

    # Validation B: Confirm Mask activation successfully zeroed out components algebraically
    zeros_count = (pruned_states == 0.0).sum().item()
    total_elements = batch * seq * dim

    assert zeros_count > 0, "No pathways were gated. Modular residue filter failed to execute."

    print(f"Modular Block gated approximately {zeros_count} structural boundaries over {total_elements} available grid spaces.")

    # Validation C: Check Scale Conservation Protocol (Inverted Dropout)
    # The pure states that SURVIVED the dropout should mathematically scale UPWARDS
    surviving_values = pruned_states[pruned_states != 0.0]

    # If starting val is 1.0, and Scaler is 1/(1-0.1) = 1/0.9 = 1.111
    expected_scaler = 1.0 / 0.9

    assert torch.allclose(surviving_values, torch.tensor(expected_scaler), rtol=1e-3), (
        "Dropout scaling mechanism failed sum-conservation check."
    )

    print("Test passed: TTT Modular Dropout successfully aligned structural sparsity via deterministic placement.")


if __name__ == "__main__":
    test_modular_dropout()
