# Security Policy

## Supported versions

Only the latest beta pre-release is supported with security fixes:

| Version   | Supported          |
| --------- | ------------------ |
| 0.1.0b1   | :white_check_mark: |
| < 0.1.0b1 | :x:                |

## Reporting a vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Report suspected vulnerabilities privately using one of these channels:

1. **GitHub private vulnerability reporting** (preferred):
   [Security → Report a vulnerability](https://github.com/Nexus-Resonance-Codex/Ai-Enhancements/security/advisories/new)
   on the repository.
2. **Email**: `NexusResonanceCodex@gmail.com` with the subject
   `[SECURITY] nrc-ai`.

Please include:

- A description of the vulnerability and its potential impact.
- Steps to reproduce (a minimal proof of concept is ideal).
- The affected version(s) and your environment (Python version, OS, torch version).
- Any suggested mitigation, if you have one.

## What to expect

- **Acknowledgment** of your report within 7 days.
- A **triage assessment** (severity, affected scope) within 14 days.
- Coordinated disclosure: we will work with you on a fix and a release
  before any public disclosure, and credit you in the release notes
  unless you prefer to remain anonymous.

## Scope

This policy covers the `nrc_ai` Python package in `src/nrc_ai/`, its build
and packaging configuration, and the CI workflows under `.github/workflows/`.
The bundled LaTeX proofs, demo assets, and wiki pages are documentation and
are out of scope, though reports about them are still welcome.
