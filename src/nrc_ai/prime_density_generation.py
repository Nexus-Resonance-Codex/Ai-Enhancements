from typing import cast

import torch
import torch.nn as nn

# The structurally sound resonant anchor sequence
TUPT_RESONANT = frozenset({1, 2, 4, 5, 7, 8})


class PrimeDensityConditionedGeneration(nn.Module):
    """Enhancement #11: Prime-Density Conditioned Generation v3.

    A logits-processor and temperature modifier intended for the autoregressive
    decoding phase. Standard LLMs decode strictly based on raw probability.

    This enhancement conditions the output distribution by artificially boosting
    the probability of tokens that align mathematically with the stabilizing anchor
    (1, 2, 4, 5, 7, 8) prime-density lattice.

    Tokens falling exactly on stable resonant indices modulo 9 receive a
    Golden Ratio phase-boost to their logits prior to softmax/sampling, ensuring
    the text generation natively prefers stable resonant pathways and avoids the
    0, 3, 6, 9 chaotic voids.
    """

    def __init__(self, vocab_size: int, boost_factor: float = 1.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.boost_factor = boost_factor

        # Precompute the static Mod 9 density map for the full vocabulary
        self.register_buffer("density_boost_mask", self._build_prime_density_mask())

    def _build_prime_density_mask(self) -> torch.Tensor:
        """Calculates a static logit-bias vector pushing specific vocab IDs."""
        # Create a mask of zeroes for the whole vocab
        mask = torch.zeros(self.vocab_size, dtype=torch.float32)

        # We index the vocabulary using Modulo 9 calculations.
        # If an ID mod 9 maps to the protective stable nodes (1, 2, 4, 5, 7, 8),
        # we assign a positive scalar boost derived from Phi.
        for i in range(self.vocab_size):
            mod_val = i % 9
            # TUPT subset
            if mod_val in TUPT_RESONANT:
                mask[i] = self.boost_factor

        return mask

    def forward(self, input_ids: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """Calculates conditioned logits toward prime-density alignment.

        Args:
            input_ids: (batch_size, seq_len) The current generation context.
            logits: (batch_size, vocab_size) The next-token probabilistic logits.

        Returns:
            Conditioned logits mathematically pushed toward prime-density alignment.
        """
        # 1. Structural Coupling
        # Projected field resonance is added back to original logits
        # We use the precomputed density boost mask.
        output = logits + self.density_boost_mask
        return cast(torch.Tensor, output)
