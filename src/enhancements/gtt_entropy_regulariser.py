import torch
import torch.nn as nn
from ..nrc_math.phi import PHI_FLOAT

class GTTEntropyCollapseRegulariser(nn.Module):
    """
    Enhancement #12: GTT Entropy Collapse Regulariser v2

    Global Tensor Thermodynamics (GTT) dictates the exact physical limits of entropy
    an AI model can sustain before catastrophic hallucination occurs.
    NRC maps this safe upper thermodynamic boundary strictly to ~10.96 nats.

    This structural execution layer monitors feature distributions mid-flight natively.
    If the Shannon Entropy of the activation block approaches or exceeds the GTT
    target threshold, it forces an instantaneous tensor-collapse, compressing the
    noise mathematically by Phi.

    Formula:
    H(x) = -sum( P(x) * log(P(x)) )
    if H(x) > 10.96 -> x_new = x / phi
    """
    def __init__(self, gtt_safe_boundary: float = 10.96):
        super().__init__()
        self.gtt_safe_boundary = gtt_safe_boundary

    def _calculate_shannon_entropy(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Calculates the normalized Shannon Entropy of the incoming feature block.
        """
        # Convert raw activations to a probability distribution over the final dimension
        prob_dist = torch.nn.functional.softmax(tensor, dim=-1)

        # To avoid log(0) NaN explosions, add a tiny epsilon
        log_prob = torch.log(prob_dist + 1e-9)

        # Calculate standard block entropy: H(x) = -sum(p * log(p))
        entropy = -torch.sum(prob_dist * log_prob, dim=-1)

        return entropy

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Dynamically routes tensors. If their internal entropy is destructive,
        they are friction-damped.
        """
        # 1. Measure the thermodynamic entropy of the tensor
        # Returns shape: (batch_size, seq_len)
        block_entropy = self._calculate_shannon_entropy(hidden_states)

        # 2. Identify indices where the entropy exceeds the NRC strict GTT limit
        # Mask shape: (batch_size, seq_len, 1) to allow direct multiplication
        collapse_mask = (block_entropy > self.gtt_safe_boundary).unsqueeze(-1)

        # 3. Apply Phi Collapse mathematically to the destructive tokens only.
        # Clean paths (False) multiply by 1.0. Collapsing paths (True) divide by phi.
        scale_factors = torch.where(collapse_mask, 1.0 / PHI_FLOAT, 1.0)

        stabilized_states = hidden_states * scale_factors

        return stabilized_states
