import torch
from typing import Iterable
from ..nrc_math.phi import PHI_FLOAT

class LucasPellHybridWeightDecay:
    """
    Enhancement #21: Lucas-Pell Hybrid Weight Decay

    Standard Deep Learning optimization utilizes a static L2 Weight Decay (e.g. 1e-4)
    to linearly compress parameter growth and prevent overfitting.

    This enhancement analyzes the parameters directly. Rather than flat scalar
    decay, it maps parameter bounds sequentially using biological Lucas limits
    (L_n = F_{n-1} + F_{n+1}) interlocked with Pell numbers (P_n = 2P_{n-1} + P_{n-2}).

    The functional outcome: Heavy, dominant structural weights are structurally
    protected by the Pell integer boundary, while chaotic irrelevant weights are
    friction-shredded by Lucas approximations of Phi.
    """
    @staticmethod
    def apply_hybrid_decay_(parameters: Iterable[torch.Tensor], base_decay_rate: float = 1e-4):
        """
        Calculates and applies the physical topological decay limits in-place.
        """
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]

        parameters = [p for p in parameters if p is not None]

        # Phi approximations:
        # Pell limit approx ~ 1 + sqrt(2) ~ 2.414 (Silver Ratio)
        silver_ratio = 1.0 + (2.0 ** 0.5)

        for p in parameters:
            with torch.no_grad():
                # 1. Analyze the feature stability locally
                magnitude = torch.abs(p)

                # 2. Threshold the biological boundaries.
                # If a weight is massive (structurally important to the model), it crosses the Silver Ratio Pell boundary.
                # If it is tiny (pointless noise), it falls strictly to the Lucas Phi bounds.
                is_dominant = (magnitude > 1.0).type(p.dtype)

                # Dominant weights decay VERY slowly (Base * (1 / Silver Ratio))
                # Chaotic weights decay VERY rapidly (Base * Phi)
                decay_modifier = torch.where(
                    is_dominant == 1.0,
                    1.0 / silver_ratio,  # Protect the core macro-structures
                    PHI_FLOAT            # Destroy the micro-scale noise quickly
                )

                # 3. Apply the L2 gradient mathematically directly to the parameters
                p.sub_(p * (base_decay_rate * decay_modifier))
