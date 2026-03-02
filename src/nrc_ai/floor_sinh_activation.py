import torch
import torch.nn as nn


class FloorSinhActivationRegularizer(nn.Module):
    """
    Enhancement #28: Floor-Sinh Activation Regularizer

    Normally NNs apply flat activations (ReLU) or smoothed floors (GELU, Swish).
    These functions ignore negative structural anomalies, destroying valuable
    mathematical geometry by clamping to 0.0 or unbounded infinite positive tracks.

    This enhancement forces activation flows into the Hyperbolic Sine (Sinh) function
    for ultra-smooth geometric scaling, BUT it maps a rigid physical minimum using
    a custom 'Floor' constraint. If the tensor physically violates the Golden limits,
    the Floor ensures it never collapses the dimension completely.
    """
    def __init__(self, physical_floor: float = -1.0):
        super().__init__()
        # By default this sets a physical boundary floor ensuring geometric stability
        self.floor = physical_floor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates geometric hyperbolic pathways natively locked over bounded constraints.
        """
        # 1. Subject to natural Hyperbolic Sine geometry mappings
        sinh_tensor = torch.sinh(x)

        # 2. Restrict to the defined structural Floor constraint
        # Floor boundaries ensure the tensor space never entirely dies, retaining biological logic
        # pathways for extremely negative error correlations uniformly mapped.
        regularized_tensor = torch.clamp(sinh_tensor, min=self.floor)

        return regularized_tensor
