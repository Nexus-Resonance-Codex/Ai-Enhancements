import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from enhancements.prime_density_generation import PrimeDensityGenerator

def test_prime_density_generation():
    """
    Validates Enhancement #11: Prime-Density Generator successfully alters
    logit distributions conditionally based on Mod 2187 TUPT sequence mappings.
    """
    vocab_size = 32000
    batch = 2

    # Simulate a raw, perfectly uniform logit output from a model (all zeros before softmax = all probabilities equal)
    raw_logits = torch.zeros(batch, vocab_size)

    generator = PrimeDensityGenerator(vocab_size=vocab_size, boost_factor=5.0)

    # 1. Forward Pass to condition the logits
    conditioned_logits = generator(None, raw_logits)  # input_ids not strictly needed for this global static mask

    # 2. Validation
    # TUPT sequence base integers: 3, 6, 9, 7
    # Logits exactly at positions [3], [6], [7], [9] should be heavily boosted (+5.0)
    assert conditioned_logits[0, 3].item() == 5.0, "TUPT Token 3 failed to receive Prime-Density Boost."
    assert conditioned_logits[0, 7].item() == 5.0, "TUPT Token 7 failed to receive Prime-Density Boost."

    # A random baseline token (e.g. 5) that isn't connected to [3,6,9,7] should remain unaltered (0.0)
    assert conditioned_logits[0, 5].item() == 0.0, "Non-resonant token illegally received Prime-Density Boost."

    # Softmax check (The probability of token 3 should now mathematically eclipse token 5 prior to sampling)
    probs = torch.nn.functional.softmax(conditioned_logits, dim=-1)
    assert probs[0, 3] > probs[0, 5], "Softmax mapping failed; boost did not manipulate probability."

    print("Test passed: Prime-Density Conditioned Logit Generator dynamically shaped vocab probabilites to resonant paths.")

if __name__ == "__main__":
    test_prime_density_generation()
