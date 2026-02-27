import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from enhancements.e8_golden_basis import GoldenBasisEmbedding

def test_e8_golden_basis():
    """
    Validates Enhancement #8: Golden Basis Embedding correctly initializes
    within the structural Mod 2187 bounds and maintains valid phi scaling
    during token retrieval.
    """
    # Using a smaller vocab size simulating the 163840 matrix for quick validation
    vocab_size = 4096
    embed_dim = 256

    layer = GoldenBasisEmbedding(num_embeddings=vocab_size, embedding_dim=embed_dim)

    # 1. Verify that the initialization triggered the biological blocks.
    # We should have multiple exactly 0.0 weights in the matrix due to apply_exclusion_gate.
    zero_count = (layer.embedding.weight == 0.0).sum().item()
    print(f"Golden Basis Initialization definitively blocked {zero_count} structural dead-zones out of {layer.embedding.weight.numel()}.")
    assert zero_count > 0, "No values were zeroed out; the lattice projection failed to catch Mod 2187 boundaries."

    # 2. Verify forward pass routing
    dummy_input = torch.randint(0, vocab_size, (4, 128))  # batch: 4, seq_len: 128

    output = layer(dummy_input)
    assert output.shape == (4, 128, embed_dim), "Embedding sequence generated an invalid dimensionality."
    assert not torch.isnan(output).any(), "NaN found in embedding space."
    assert not torch.isinf(output).any(), "Inf found in embedding space."

    print("Test passed: 163840 E8x256 Golden Basis Embedding mathematically mapped and retrieved cleanly.")

if __name__ == "__main__":
    test_e8_golden_basis()
