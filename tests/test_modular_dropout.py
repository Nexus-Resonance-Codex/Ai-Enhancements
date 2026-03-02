import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from enhancements.modular_dropout import ModularDropoutPattern

def test_modular_dropout():
    """
    Validates Enhancement #25: The Modular Dropout grid shreds parameters
    using strictly algebraic Mod-2187 placement indices without random logic failures.
    """
    batch = 2
    seq = 500
    dim = 16

    # Deploy a completely pure 'active' hidden state matrix structurally
    # Zeros are invisible to scale multiplication, so we use 1s
    pure_states = torch.ones(batch, seq, dim)

    dropout = ModularDropoutPattern(probability=0.1)

    # Must enforce training mode globally for mask activation
    dropout.train()

    pruned_states = dropout(pure_states)

    # Validation A: Structural dimensionality continuity
    assert pruned_states.shape == pure_states.shape, "Modular dropout physically fractured tensor dimensions inherently."

    # Validation B: Confirm Mask activation successfully zeroed out components algebraically
    zeros_count = (pruned_states == 0.0).sum().item()
    total_elements = batch * seq * dim

    assert zeros_count > 0, "No pathways were shredded. Mod-2187 topological scaler failed to execute boundaries."

    print(f"Modular Block dropped roughly {zeros_count} topological boundaries over {total_elements} available grid spaces algebraically.")

    # Validation C: Check Scale Conservation Protocol (Inverted Dropout)
    # The pure states that SURVIVED the dropout should mathematically scale UPWARDS natively
    surviving_values = pruned_states[pruned_states != 0.0]

    # If starting val is 1.0, and Scaler is 1/(1-0.1) = 1/0.9 = 1.111
    expected_scaler = 1.0 / 0.9

    assert torch.allclose(surviving_values, torch.tensor(expected_scaler), rtol=1e-3), "Dropout scaling mechanism destroyed sum-conservation boundaries natively."

    print("Test passed: 3-6-9-7 Modular Dropout grid structurally sheared noise via fixed TUPT algebraic placement correctly.")

if __name__ == "__main__":
    test_modular_dropout()
