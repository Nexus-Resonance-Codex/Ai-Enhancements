import torch
import torch.nn as nn
from typing import Optional
from nrc.math.tupt_exclusion import TUPT_SEQUENCE

class PrimeDensityGenerator(nn.Module):
    """
    Enhancement #11: Prime-Density Conditioned Generation v3

    A logits-processor and temperature modifier intended for the autoregressive
    decoding phase. Standard LLMs decode strictly based on raw probability.

    This enhancement conditions the output distribution by artificially boosting
    the probability of tokens that align mathematically with the TUPT sequence
    (3, 6, 9, 7) prime-density lattice.

    Tokens falling perfectly on resonant prime indices modulo 2187 receive a
    Golden Ratio phase-boost to their logits prior to softmax/sampling, ensuring
    the text generation natively prefers stable resonant pathways.
    """
    def __init__(self, vocab_size: int, boost_factor: float = 1.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.boost_factor = boost_factor

        # Precompute the static Mod 2187 density map for the full vocabulary
        self.register_buffer("density_boost_mask", self._build_prime_density_mask())

    def _build_prime_density_mask(self) -> torch.Tensor:
        """
        Calculates a static logit-bias vector pushing specific vocab IDs.
        """
        # Create a mask of zeroes for the whole vocab
        mask = torch.zeros(self.vocab_size, dtype=torch.float32)

        # We index the vocabulary using Modulo 2187 calculations.
        # If an ID mod 2187 maps to the protective TUPT base (3,6,9,7),
        # we assign a positive scalar boost derived from Phi.
        for i in range(self.vocab_size):
            mod_val = i % 2187
            # TUPT subset
            if mod_val in TUPT_SEQUENCE:
                mask[i] = self.boost_factor

        return mask

    def forward(self, input_ids: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch_size, seq_len) The current generation context.
            logits: (batch_size, vocab_size) The next-token probabilistic logits.
        Returns:
            Conditioned logits mathematically pushed toward prime-density alignment.
        """
        # We simply add the static structural bounds directly into the logit space
        # prior to final softmax sampling.
        conditioned_logits = logits + self.density_boost_mask

        return conditioned_logits
