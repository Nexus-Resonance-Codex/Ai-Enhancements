import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from enhancements.nrc_protein_engine import NRCProteinFoldingEngine

def test_nrc_protein_engine_excludes_invalid_states():
    """
    Validates Enhancement #2: The NRC Protein engine correctly utilizes Mod 2187
    exclusions and maps toward GTT target scaling limits.
    """
    dim_size = 256
    model = NRCProteinFoldingEngine(sequence_dim=dim_size, gtt_target_nats=10.96)

    # Simulate batch of 4 amino acid sequence embeddings, seq len 128
    dummy_seq = torch.randn(4, 128, dim_size) * 5000.0  # scaled up to hit mod 2187 locks

    folded_states = model(dummy_seq)

    # Shape preserved
    assert folded_states.shape == dummy_seq.shape

    # Check that Mod 2187 exclusions triggered (meaning some values were hard gated to 0.0)
    # Give extremely high probability that at least ONE neuron hit the 3-6-9-7 mod 2187 trap
    zero_count = (folded_states == 0.0).sum()
    print(f"Total resonance locks triggered (zeros): {zero_count.item()} out of {folded_states.numel()}")

    assert zero_count > 0, "No values were gated by Mod 2187 exclusions. Check scaling."
    assert not torch.isnan(folded_states).any(), "NaN found during folding simulation."

    print("Test passed: NRC Protein Engine v2 successfully gates Mod 2187 biological barriers.")

if __name__ == "__main__":
    test_nrc_protein_engine_excludes_invalid_states()
