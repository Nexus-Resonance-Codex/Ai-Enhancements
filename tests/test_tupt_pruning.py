import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from nrc_ai.tupt_token_pruning import TUPTExclusionTokenPruning


def test_tupt_token_pruning() -> None:
    """Validates Enhancement #22: TUPT Modular State Pruning.

    Optimizes processing complexity by identifying and gating tokens that align
    with unstable modular residue classes in the TUPT domain.
    """
    batch_size = 1
    seq_len = 100
    embed_dim = 16

    # Simulate a standard transformer batch holding 100 tokens
    hidden_states = torch.randn(batch_size, seq_len, embed_dim)

    # Initialize the modular residue filter
    pruner = TUPTExclusionTokenPruning()

    pruned_states = pruner(hidden_states)

    # 1. Verification of structural dimensionality properties
    # The embed_dim must remain uncorrupted
    assert pruned_states.shape[2] == embed_dim, "Token Pruning corrupted embedding representation depth."

    # 2. Verification of mathematical optimization properties
    # The output seq_len should be smaller than input seq_len due to stability gating.
    survived_seq_len = pruned_states.shape[1]

    print(f"Original Sequence Context: {seq_len} tokens.")
    print(f"Stable Sequence Context:   {survived_seq_len} tokens.")

    assert survived_seq_len < seq_len, "TUPT failed to trim tokens based on stability nodes."

    print("Test passed: TUPT Modular State Pruning successfully optimized inference context via stability gating.")


if __name__ == "__main__":
    test_tupt_token_pruning()
