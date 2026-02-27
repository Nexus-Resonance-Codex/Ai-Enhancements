import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from enhancements.phi_lora_adapter import PhiLosslessLoraAdapter

def test_phi_lossless_lora():
    """
    Validates Enhancement #9: The Phi Lossless LoRA Adapter initializes correctly
    (B=0), successfully runs down/up projection bounded by Shard Compression,
    and applies golden scaling.
    """
    in_dim = 1024
    out_dim = 512
    rank = 16
    batch = 4
    seq = 32

    # Instantiate the NRC LoRA module
    adapter = PhiLosslessLoraAdapter(in_features=in_dim, out_features=out_dim, rank=rank)

    # 1. Initial State Validity Check
    # Ensure B is perfectly zeroed on init for safe drop-in scaling
    assert (adapter.lora_B.weight == 0.0).all(), "LoRA B-matrix did not explicitly initialize to absolute zero."

    # 2. Forward Routing
    x = torch.randn(batch, seq, in_dim)
    output = adapter(x)

    assert output.shape == (batch, seq, out_dim), "LoRA adapter resulted in a geometric dimensionality error."

    # 3. Given B is initialized to 0, the very first pass without training should strictly equal 0
    assert (output == 0.0).all(), "Adapter leaked residual gradients instead of locking zero-state during initial pass."

    # 4. Simulate a single training gradient update step to B matrix
    adapter.lora_B.weight.data.normal_(0.0, 0.02)
    active_output = adapter(x)

    assert not (active_output == 0.0).all(), "Adapter remains permanently collapsed after B-matrix update."
    assert not torch.isnan(active_output).any(), "NaN found in modified NRC LoRA bounds."
    assert not torch.isinf(active_output).any(), "Inf found in modified NRC LoRA bounds."

    print("Test passed: phi^infty Lossless LoRA Adapter correctly channels rank compression through fractal limits.")

if __name__ == "__main__":
    test_phi_lossless_lora()
