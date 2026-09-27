# Changelog

All notable changes to the `nrc-ai` package are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0b1] - Unreleased

First public beta of the hardened `nrc-ai` build: the 30+ module Nexus
Resonance Codex AI-enhancement suite packaged as an installable, CI-gated,
pure open-source (AGPL-3.0) Python library.

### Added

- `nrc-ai` package layout under `src/nrc_ai/` with a public `__init__`
  re-exporting all 30+ enhancement modules (attention & positional encoding,
  context memory & KV-cache compression, optimizers & schedulers, layers,
  encodings & activations).
- Packaging metadata via `hatchling` (`pyproject.toml`), including the
  `dev` extra with `pytest`, `pytest-cov`, `hypothesis`, `ruff`, and `mypy`.
- Full verification suite under `tests/` covering module numerics and
  stability invariants.
- Hardened CI: dedicated lint workflow (`ruff check`, `ruff format --check`,
  `mypy`), a `pytest` matrix across Python 3.10–3.13, an aggregate
  `ci-success` branch-protection gate, and a tag-triggered PyPI release
  workflow using trusted publishing (OIDC, no stored secrets).
- Community files: issue forms (bug report, feature request), PR template,
  `SECURITY.md`, `CITATION.cff`, `CONTRIBUTING.md`, and this changelog.
- MkDocs documentation site (`docs/`) and the module wiki pages (`wiki/`).

### Changed

- Repository aligned to a single open-source license posture: code under
  AGPL-3.0, data & weights under CC BY-NC-SA 4.0. All commercial-licensing
  language and the `COMMERCIAL_USE.md` file were removed.

### Fixed

- Mypy target aligned to Python 3.12 and CI install paths consolidated onto
  `uv`/`pip` editable installs of the package with its `dev` extras.

[0.1.0b1]: https://github.com/Nexus-Resonance-Codex/Ai-Enhancements/releases/tag/v0.1.0b1
