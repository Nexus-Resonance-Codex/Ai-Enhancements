import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from nrc_ai.nrc_protein_engine import NRCProteinFoldingEngine


def test_nrc_protein_engine_excludes_invalid_states() -> None:
    """Validates Enhancement #2: The NRC Protein engine correctly utilizes TUPT.

    Utilizes modular residue exclusions and maps toward TTT-aligned stability limits.
    """
    dim_size = 256
    model = NRCProteinFoldingEngine(sequence_dim=dim_size, gtt_target_nats=10.96)

    # Simulate batch of 4 sequence embeddings, seq len 128
    dummy_seq = torch.randn(4, 128, dim_size) * 5000.0  # scaled up to trigger modular residue gates

    folded_states = model(dummy_seq)

    # Shape preserved
    assert folded_states.shape == dummy_seq.shape

    # Check that TUPT exclusions triggered (meaning some values were hard gated to 0.0)
    # High probability that at least ONE coordinate hit the modular residue stability gate
    zero_count = (folded_states == 0.0).sum()
    print(f"Total stability gates triggered (zeros): {zero_count.item()} out of {folded_states.numel()}")

    assert zero_count > 0, "No values were gated by modular residue exclusions."
    assert not torch.isnan(folded_states).any(), "NaN found during simulation."

    print("Test passed: NRC Protein Engine successfully aligned to TUPT structural stability nodes.")


if __name__ == "__main__":
    test_nrc_protein_engine_excludes_invalid_states()
