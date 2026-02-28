import torch
import torch.nn as nn
from torch.autograd import Function
from nrc.math.tupt_exclusion import apply_exclusion_gate
from nrc.math.phi import PHI_FLOAT

class ExclusionGradientRouterFunction(Function):
    """
    Core Autograd function for Enhancement #6.

    Forward pass: Acts as a sparse gate, blocking activations that hit the
    TUPT Mod 2187 exclusion zones.

    Backward pass: Gradients flowing back *also* pass through the exclusion mask.
    Surviving gradients mathematically accelerate by phi (golden ratio) to maintain
    total system resonance and Entropy Target.
    """
    @staticmethod
    def forward(ctx, inputs):
        # Apply the exact Mod 2187 gate from NRC
        # apply_exclusion_gate returns 0.0 for elements triggering the biological lockout
        routed_inputs = apply_exclusion_gate(inputs)

        # Save the mask itself (1.0 for pass, 0.0 for blocked) to use in gradient routing
        mask = (routed_inputs != 0.0).type(inputs.dtype)
        ctx.save_for_backward(mask)

        return routed_inputs

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors

        # 1. Biological Gradient Exclusion
        # Gradients matching blocked forward paths are zeroed out (dead pathways)
        routed_grad = grad_output * mask

        # 2. Golden Gradient Acceleration
        # Paths that *survived* are mathematically amplified by phi to concentrate
        # flow along resonant trajectories
        resonant_grad = routed_grad * PHI_FLOAT

        return resonant_grad

class BiologicalExclusionGradientRouter(nn.Module):
    """
    Enhancement #6: Biological Exclusion Gradient Router v3

    A structural layer designed to be placed between deep network blocks.
    It functions as an advanced non-linear Dropout/MoE Router alternative.
    Rather than dropping randomly, it drops exactly the values strictly forbidden
    by the Nexus Resonance Codex (TUPT Mod 2187 exclusions), and amplifies back-prop
    error signals along surviving golden paths.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ExclusionGradientRouterFunction.apply(x)
