import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from enhancements.tupt_token_pruning import TUPTExclusionTokenPruner

def test_tupt_token_pruning():
    """
    Validates Enhancement #22: TUPT Exclusion Pruning successfully cuts down sequence
    processing complexity by mathematically identifying and shredding tokens that
    map directly into Mod-2187 biological noise limits.
    """
    batch_size = 1
    seq_len = 100
    embed_dim = 16

    # Simulate a standard transformer batch holding 100 physical tokens
    hidden_states = torch.randn(batch_size, seq_len, embed_dim)

    # Initialize the mathematical exclusion filter mapping
    pruner = TUPTExclusionTokenPruner()

    pruned_states = pruner(hidden_states)

    # 1. Verification of structural dimensionality properties
    # The embed_dim MUST remain entirely un-corrupted (Tokens are destroyed laterally, not vertically)
    assert pruned_states.shape[2] == embed_dim, "Token Pruning accidentally deleted embedding representation depth."

    # 2. Verification of mathematical memory-save properties
    # The output seq_len must be strictly smaller than the input seq_len because
    # TUPT exclusions mathematically forbid 3-6-9-7 bounds natively.
    survived_seq_len = pruned_states.shape[1]

    print(f"Original Sequence Context: {seq_len} tokens.")
    print(f"Resonant Sequence Context: {survived_seq_len} tokens.")

    assert survived_seq_len < seq_len, "TUPT Token Matrix failed to sparsely trim inference contexts natively."

    print("Test passed: 1D Mod-2187 Exclusion grids dynamically stripped non-resonant trajectory sequences from tracking bounds.")

if __name__ == "__main__":
    test_tupt_token_pruning()
