"""Tests for Ai-Enhancements core modules."""

import torch

from nrc_ai import ExecutiveAgent, PhiInfinityPersistentMemory, PhiInfinityShardFolding


def test_shard_folding_forward() -> None:
    """Verify shard folding output shape and signs."""
    folder = PhiInfinityShardFolding(k_steps=2)
    x = torch.tensor([-1.5, 2.7, 0.0, -0.1])
    out = folder(x)
    assert out.shape == x.shape
    # Check that signs are preserved for non-zero values
    mask = x != 0
    assert torch.all(torch.sign(out[mask]) == torch.sign(x[mask]))


def test_persistent_memory_update() -> None:
    """Verify persistent memory update cycle."""
    mem = PhiInfinityPersistentMemory(hidden_dim=128)
    initial_state = mem.lattice_state.clone()

    update_vec = torch.randn(1, 128)
    new_state = mem.update(update_vec)

    assert not torch.allclose(initial_state, new_state)
    assert new_state.shape == (1, 128)
    assert torch.allclose(mem.lattice_state, new_state)


def test_executive_agent_spawn() -> None:
    """Verify executive agent sub-model spawning."""
    agent = ExecutiveAgent("Nexus-Alpha")
    sub = agent.spawn_sub_model("Optimize Lattice Resonance")

    assert sub["parent"] == "Nexus-Alpha"
    assert "Nexus-Alpha-SubModel-" in sub["agent"]
    assert sub["resonance"] == "STABILIZED"


def test_metadata() -> None:
    """Verify package metadata."""
    from nrc_ai import __about__

    assert __about__.__version__ == "1.0.0"
    assert __about__.__author__ == "James Trageser"
