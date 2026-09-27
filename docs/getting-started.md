# Getting Started

This guide gets `nrc-ai` installed and running in a few minutes.

## Requirements

- Python 3.10 or newer (3.10, 3.11, 3.12 supported; 3.13 experimental)
- PyTorch 2.2 or newer

## Install

The recommended path is an editable install from the repository, which also
pulls in the NRC core library:

```bash
# Clone the repository
git clone https://github.com/Nexus-Resonance-Codex/Ai-Enhancements.git
cd Ai-Enhancements

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the package (CPU PyTorch wheels work fine for exploration)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

For development (tests, lint, type checks), install the dev extras instead:

```bash
pip install -e ".[dev]"
```

See [CONTRIBUTING.md](../CONTRIBUTING.md) for the full contributor workflow
(`pytest`, `ruff`, `mypy`).

## Quickstart

```python
import torch
from nrc_ai import (
    GoldenFlowNorm,
    GoldenSpiralRotaryEmbedding,
    HodgePhiTTorsionAttention,
    PhiInfinityLosslessLoRA,
    PisanoModulatedLRSchedule,
)

dim, heads = 512, 8

# Drop-in NRC replacements for standard transformer components
norm = GoldenFlowNorm(dim)
attn = HodgePhiTTorsionAttention(embed_dim=dim, num_heads=heads)
rope = GoldenSpiralRotaryEmbedding(dim=dim // heads)

# Example: a single forward pass through attention
x = torch.randn(2, 64, dim)          # (batch, seq_len, embed_dim)
queries = rope(x)                     # golden-spiral rotary positions
out = attn(norm(x), queries)          # φ-torsion stabilized attention
print(out.shape)                      # torch.Size([2, 64, 512])

# Example: φ-lossless LoRA adapter on a linear layer
linear = torch.nn.Linear(dim, dim)
adapter = PhiInfinityLosslessLoRA(linear, rank=8)
print(adapter(torch.randn(2, 64, dim)).shape)

# Example: Pisano-modulated learning-rate schedule
opt = torch.optim.AdamW([torch.nn.Parameter(torch.randn(dim))], lr=1e-3)
sched = PisanoModulatedLRSchedule(opt)
sched.step()
print(sched.get_last_lr())
```

## Next steps

- Browse the full module taxonomy in the [README](../README.md#architectural-taxonomy).
- Read the module guides in the [wiki](https://github.com/Nexus-Resonance-Codex/Ai-Enhancements/wiki).
- Run the test suite with `pytest tests/` to verify your environment.
- See the [Ollama guide](OLLAMA_GUIDE.md) for running NRC models locally.
