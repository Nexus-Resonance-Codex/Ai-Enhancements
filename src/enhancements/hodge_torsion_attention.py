import torch
import torch.nn as nn
import math
from ..nrc_math.phi import PHI_FLOAT

class HodgeTorsionAttention(nn.Module):
    """
    Enhancement #7: Hodge-φ^T Torsion Attention v3

    A structural upgrade to standard Multi-Head Attention (MHA) or Scaled
    Dot-Product Attention. Standard attention purely uses the dot product (Q·K^T).

    NRC geometric theory dictates that spatial information routing is strictly
    enhanced by introducing a "torsion" or geometric skew bounded by the
    Golden Ratio (φ).

    Formula:
    Attention(Q, K, V) = softmax( (Q·K^T + φ^T_torsion) / sqrt(d) ) * V
    Where φ^T_torsion is a deterministically rotating matrix embedding derived
    from the Giza slope (51.85 degrees).
    """
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads."
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # NRC Giza slope mapped to radians: 51.85 * (pi / 180) ~ 0.9049
        # The torsion scalar dictates the "twist" amplitude applied across the diagonal
        self.giza_torsion_radians = 51.85 * (math.pi / 180.0)

        # Structural Q, K, V projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def _generate_torsion_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Generates the phi-weighted torsion matrix dynamically based on sequence bounds.
        The torsion matrix applies a sinusoidal phase-twist scaled by phi across
        positional relationships.
        """
        # Create positional grid (seq_len x seq_len)
        position_indices = torch.arange(seq_len, device=device, dtype=torch.float32)
        relative_positions = position_indices.unsqueeze(0) - position_indices.unsqueeze(1)

        # Apply the Giza twist angle and the Phi constant
        # phi_torsion = phi * sin(theta_giza * relative_distance)
        torsion_bias = PHI_FLOAT * torch.sin(self.giza_torsion_radians * relative_positions)

        # We broadcast across batch and num_heads: (1, 1, seq_len, seq_len)
        return torsion_bias.unsqueeze(0).unsqueeze(0)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch_size, seq_len, embed_dim)
            attention_mask: Optional boolean or float mask
        """
        batch_size, seq_len, _ = hidden_states.shape

        # 1. Project Q, K, V
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 2. Standard Dot Product (batch, heads, seq, seq)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

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

        return self.out_proj(attn_output)
