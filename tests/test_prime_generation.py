import torch

from nrc_ai.prime_density_generation import PrimeDensityConditionedGeneration


def test_prime_density_generation() -> None:
    """Validates Enhancement #11: Prime-Density Generator correctly pulses stable IDs."""
    vocab_size = 32000
    generator = PrimeDensityConditionedGeneration(vocab_size=vocab_size, boost_factor=5.0)

    # 1. Simulate uniform logits
    raw_logits = torch.zeros(1, vocab_size)
    conditioned = generator(None, raw_logits)

    # 2. Resonant ID Verification (Stable: 1, 2, 4, 5, 7, 8 mod 9)
    # ID 1 (1 mod 9) should be boosted
    assert conditioned[0, 1].item() == 5.0
    # ID 7 (7 mod 9) should be boosted
    assert conditioned[0, 7].item() == 5.0

    # 3. Chaotic ID Verification (Excluded: 0, 3, 6, 9 mod 9)
    # ID 3 (3 mod 9) should NOT be boosted (Entropy Floor)
    assert conditioned[0, 3].item() == 0.0, "Chaotic token 3 was incorrectly boosted!"
    # ID 0 (0 mod 9) should NOT be boosted
    assert conditioned[0, 0].item() == 0.0

    # Softmax check (The probability of token 7 should now mathematically eclipse token 3)
    probs = torch.nn.functional.softmax(conditioned, dim=-1)
    assert probs[0, 7] > probs[0, 3], "Softmax mapping failed; boost did not manipulate probability."
