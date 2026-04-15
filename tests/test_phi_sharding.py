import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from nrc_ai.phi_sharding_compression import PhiShardingCompression


def test_phi_sharding_compression() -> None:
    """Validates Enhancement Core: Phi Sharding Compression Matrix."""
    input_dim = 1024
    compress_dim = 512
    phi_sharding = PhiShardingCompression(input_dim=input_dim, compress_dim=compress_dim)

    # 1. Test instantiation and matrix initialization
    assert phi_sharding.golden_matrix.shape == (compress_dim, input_dim)

    # 2. Test forward projection
    x = torch.randn(2, 8, input_dim)  # (batch, seq, dim)
    out = phi_sharding(x)
    assert out.shape == (2, 8, compress_dim)
    assert not torch.isnan(out).any()

    # 3. Verify normalization (variance preservation)
    row_norms = torch.norm(phi_sharding.golden_matrix.data, p=2, dim=1)
    assert torch.allclose(row_norms, torch.ones_like(row_norms), atol=1e-5)

    # 4. Verify exclusion of chaotic nodes (lines 29-30)
    # Check if a known chaotic column (e.g., column 0, 3, 6) is zeroed out
    assert torch.all(phi_sharding.golden_matrix[:, 0] == 0)
    assert torch.all(phi_sharding.golden_matrix[:, 3] == 0)
    assert torch.all(phi_sharding.golden_matrix[:, 6] == 0)

    print("Test passed: Phi Sharding Compression Matrix successfully projects and stabilizes high-D data.")


if __name__ == "__main__":
    test_phi_sharding_compression()
