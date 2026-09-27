# Contributing to nrc-ai

Thank you for helping improve the Nexus Resonance Codex AI Enhancements.
This is a pure open-source project: all code contributions are provided
under the project's **AGPL-3.0** license, and data/weights under
**CC BY-NC-SA 4.0**.

## Development setup

Requirements: Python 3.10+ and (recommended) [`uv`](https://docs.astral.sh/uv/).

```bash
# Clone and enter the repo
git clone https://github.com/Nexus-Resonance-Codex/Ai-Enhancements.git
cd Ai-Enhancements

# Create a virtual environment and install editable with dev tools
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Without `uv`, standard `pip` works the same way:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The dev extras install `pytest`, `pytest-cov`, `hypothesis`, `ruff`, and `mypy`.

## Checks: test, lint, format, types

Run these before opening a pull request — the CI workflows run the same commands:

```bash
# Test suite (with coverage)
pytest tests/

# Lint
ruff check src/ tests/

# Format check (apply with: ruff format src/ tests/)
ruff format --check src/ tests/

# Type check
mypy src/nrc_ai
```

Style notes:

- Line length is 150 (see `[tool.ruff]` in `pyproject.toml`).
- New modules must include docstrings in Google convention and type annotations
  on the public API.
- Mathematical claims in docstrings should reference the corresponding proof or
  derivation in `proofs/` or `docs/` where one exists.

## Adding tests

- Place tests under `tests/`; name files `test_<module>.py`.
- Cover both the happy path and the stability invariant the module claims
  (e.g. bounded gradients, non-collapsing representations).
- Property-based tests with `hypothesis` are welcome for numeric invariants.

## Pull request process

1. Fork the repository and create a feature branch from `main`
   (e.g. `feature/golden-spiral-rope-v2`). Keep branches focused: one logical
   change per PR.
2. Update documentation if the public API changes (`README.md`, `docs/`, and
   the wiki page for the module if one exists).
3. Make sure the full check suite above passes locally.
4. Open the PR against `main` and complete the PR template checklist.
5. A maintainer reviews for correctness of both code **and** the underlying
   mathematics; be prepared to point at the proof.

## Reporting bugs and requesting features

Use the issue forms in the **Issues** tab (bug report / feature request).
Feature requests should fit the project's scope: deterministic,
mathematically grounded PyTorch primitives based on NRC geometry.

## Code of conduct

Be precise, be kind, and cite your math. Disagreements about formulations are
settled with derivations, not volume.
