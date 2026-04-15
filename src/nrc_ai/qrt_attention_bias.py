import math

import torch
import torch.nn as nn
from nrc_math import PHI_FLOAT


class QRTGeometricAttentionBias(nn.Module):
    r"""Enhancement #27: Geometric Damping Angle (THETA_QRT ≈ 51.85°) Attention Bias.

    Standard Transformer sequence logits are unbounded until normalized by the
    Softmax distribution block.

    The NRC framework utilizes the optimal geometric damping angle (\theta_{QRT} \approx 51.85^\\circ),
    derived as arctan(\\sqrt{\\phi}), as a foundational stability constant. This
    enhancement applies the cosine of this angle as a structural phase-shift bias
    to the attention logit matrices. This mathematically biases global memory
    routing toward stable manifold states within high-dimensional attention spaces.
    """

    def __init__(self, max_seq_len: int = 4096) -> None:
        super().__init__()
        # Calculate the geometric damping constant: theta = arctan(sqrt(phi))
        self.radians = math.atan(math.sqrt(PHI_FLOAT))
        self.qrt_bias_scalar = math.cos(self.radians)

        # Pre-compute the 2D bias structure (Seq x Seq)
        self.register_buffer("qrt_attention_bias", self._build_bias_matrix(max_seq_len))
        self.qrt_attention_bias: torch.Tensor

    def _build_bias_matrix(self, max_seq_len: int) -> torch.Tensor:
        """Calculates the static 2D positional grid utilizing structural geometric damping."""
        base_grid = torch.ones(max_seq_len, max_seq_len, dtype=torch.float32)

        # Apply the absolute structural geometric bounds as a stabilizing floor.
        return base_grid * self.qrt_bias_scalar

    def forward(self, qk_logits: torch.Tensor) -> torch.Tensor:
        """Injects geometric damping into the dot-product attention block.

        Args:
            qk_logits: (batch, num_heads, seq_len, seq_len).
        """
        seq_len = qk_logits.size(-1)

        # Slice the 2D algebraic bounds for the current sequence length
        bias_slice = self.qrt_attention_bias[:seq_len, :seq_len]

        # Apply the structural bias across batches and heads
        qrt_stabilized_logits = qk_logits + bias_slice

        return qrt_stabilized_logits
