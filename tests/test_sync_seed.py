#  Nexus Resonance Codex - 2025-2026 Breakthrough Series
#  Copyright (c) 2026 James Trageser (@jtrag)
#
#  Licensed under CC-BY-NC-SA-4.0 + NRC-L
#  "This work is part of the Nexus Resonance Codex (NRC) incorporating TTT
#  modular exclusion, phi^inf compression, 256D->729D lattice, QRT, and MST."

"""Tests for TUPTSyncSeed module."""

import torch
from nrc_ai.tupt_sync_seed import TUPTSyncSeed


def test_tupt_sync_seed_generation() -> None:
    """Verify TUPTSyncSeed deterministic synchronization and modular stability."""
    base_multiplier = 42
    TUPTSyncSeed.synchronize(base_multiplier=base_multiplier)
    
    # Verify consistent modular stability class (e.g., Mod 9 alignment)
    # The seeder should produce seeds meeting the TTT modular residue stability criteria.
    # Since we can't easily check the global seed, we check if global state is set.
    assert torch.initial_seed() % 9 in {1, 2, 4, 5, 7, 8}


def test_tupt_sync_seed_determinism() -> None:
    """Verify seeder determinism from the same base."""
    TUPTSyncSeed.synchronize(base_multiplier=123)
    s1 = torch.initial_seed()
    TUPTSyncSeed.synchronize(base_multiplier=123)
    s2 = torch.initial_seed()
    assert s1 == s2
