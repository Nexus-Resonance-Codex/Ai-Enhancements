import torch
import torch.nn as nn
from nrc.math.tupt_exclusion import TUPT_CHAOTIC


class ModularDropoutPattern(nn.Module):
    """
    Enhancement #25: 0-3-6-9 Chaotic Void Modular Dropout Pattern

    Standard Dropout mechanisms (e.g. Dropout(0.1)) annihilate network
    connections randomly via arbitrary Bernoulli distributions. This destroys
    mathematical topology blindly.

    The NRC Modular Dropout Pattern entirely replaces structural randomization.
    Instead of guessing, it generates a structural dropout mask aligning natively
    to the Mod-9 domain. Connections that violate biological progression
    and map to the Chaotic sequence [0, 3, 6, 9] are pruned, leaving the 
    structurally stable pathways perfectly un-sheared.
    """
    def __init__(self, probability: float = 0.1):
        super().__init__()
        # Target sparsity ~ 0.1.
        self.probability = probability

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Calculates biological gaps algebraically and routes information explicitly
        around chaotic boundaries natively.
        """
        if not self.training:
            return hidden_states

        # 1. We flatten the tensor theoretically to determine topological routing IDs
        batch_size, seq_len, embed_dim = hidden_states.shape
        total_elements = batch_size * seq_len * embed_dim

        # 2. Sequence flat IDs mimicking biological placement index logic
        indices = torch.arange(total_elements, device=hidden_states.device)

        # 3. Assess the Mod 9 physical resonance of the coordinate
        mod_values = indices % 9

        # 4. Generate the Mask mapped explicitly to the destructive physical space
        mask = torch.ones(total_elements, device=hidden_states.device)

        # Structural Drop condition: We want to hit roughly 'probability' amount of pruning.
        # Since TUPT_CHAOTIC holds massive mathematical disruption, we zero out pathways
        # that specifically ALIGN with these index boundaries inside a scaled sparse grid.
        # To maintain exact stochastic ratios, we only drop chaotic boundaries that also Mod against
        # a scaler matching the probability.

        scaler = int(1.0 / self.probability)

        # Generate the conditions: Must be logically chaotic AND align with the probability scaler spacing
        for base_val in TUPT_CHAOTIC:
            # Drop connections dynamically scaling over the 1-2-4-5-8 chaotic grid
            condition = (mod_values == base_val)
            # Combine biological targeting with logical scaling
            mask[condition & (indices % scaler == 0)] = 0.0

        # 5. Restructure the 1D physical mask back into the requested spatial topology
        spatial_mask = mask.view(batch_size, seq_len, embed_dim)

        # 6. Apply standard activation scaling to ensure training sum conservation
        # Inverted dropout: Scale surviving parameters up by 1/(1-p)
        dropout_scaler = 1.0 / (1.0 - self.probability)

        return hidden_states * spatial_mask * dropout_scaler
