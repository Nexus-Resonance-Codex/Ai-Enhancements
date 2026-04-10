import math
from typing import Optional, cast

import torch
import torch.nn as nn
from nrc_math import PHI_FLOAT


class HodgePhiTTorsionAttention(nn.Module):
    """Enhancement #7: Hodge-φ^T Torsion Attention v3.

    A structural upgrade to standard Multi-Head Attention (MHA) or Scaled
    Dot-Product Attention. Standard attention purely uses the dot product (Q·K^T).

    NRC geometric theory dictates that spatial information routing is strictly
    enhanced by introducing a "torsion" or geometric skew bounded by the
    Golden Ratio (φ).

    Formula:
    Attention(Q, K, V) = softmax( (Q·K^T + φ^T_torsion) / sqrt(d) ) * V
    Where φ^T_torsion is a deterministically rotating matrix embedding derived
    from the exact tangent limit arctan(sqrt(phi)).
    """

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads."
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Optimal geometric damping angle mapped directly to analytical radians
        # The torsion scalar dictates the "skew" amplitude applied across the diagonal
        self.qrt_torsion_angle = math.atan(math.sqrt(PHI_FLOAT))

        # Structural Q, K, V projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def _generate_torsion_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generates the phi-weighted torsion matrix dynamically based on sequence bounds.

        The torsion matrix applies a sinusoidal phase-twist scaled by phi across
        positional relationships.
        """
        # Create positional grid (seq_len x seq_len)
        position_indices = torch.arange(seq_len, device=device, dtype=torch.float32)
        relative_positions = position_indices.unsqueeze(0) - position_indices.unsqueeze(1)

        # Apply the optimal geometric damping angle and the Phi constant
        # phi_torsion = phi * sin(theta_qrt * relative_distance)
        torsion_bias = PHI_FLOAT * torch.sin(self.qrt_torsion_angle * relative_positions)

        # We broadcast across batch and num_heads: (1, 1, seq_len, seq_len)
        return torsion_bias.unsqueeze(0).unsqueeze(0)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Executes the Hodge-Torsion Attention pass.

        Args:
            hidden_states: (batch_size, seq_len, embed_dim)
            attention_mask: Optional boolean or float mask.
        """
        batch_size, seq_len, _ = hidden_states.shape

        # 1. Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # 2. Reshape to multi-head (B, H, S, D)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Calculate Scaled Dot-Product Attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 3. Add Hodge-Phi Torsion Bias
        # This breaks isotropic attention isotropy geometrically
        torsion_bias = self._generate_torsion_bias(seq_len, device=hidden_states.device)
        attn_weights = attn_weights + torsion_bias

        # 4. Standard mask applicability (e.g., causal causal or padding mask)
        if attention_mask is not None:
            # Assumes mask is broadcastable to (batch, heads, seq, seq)
            attn_weights = attn_weights + attention_mask

        # 5. Softmax and V product
        attn_probs = torch.nn.functional.softmax(attn_weights, dim=-1)

        # (batch, heads, seq, dim)
        attn_output = torch.matmul(attn_probs, v)

        # 6. Re-assemble heads and project to output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.head_dim)

        return cast(torch.Tensor, self.out_proj(attn_output))
