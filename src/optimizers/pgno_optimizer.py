import math

import torch
from torch.optim.optimizer import Optimizer


class PGNOptimizer(Optimizer):
    """Prime Geometric Node Optimizer (PGNO) - 2048D NRC Framework.

    Optimizes gradients not by magnitude, but by projecting them into the
    Giza 51.827 degree manifold and avoiding the 0-3-6-9 chaotic voids.
    Gradients landing on chaotic voids are zeroed (TUPT exclusion).
    """

    def __init__(self, params, lr=1e-3, phi_momentum=0.618):
        defaults = {"lr": lr, "phi_momentum": phi_momentum}
        super(PGNOptimizer, self).__init__(params, defaults)
        self.giza_slope_rad = 51.827 * math.pi / 180.0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            phi_momentum = group["phi_momentum"]
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                # Apply Giza Slope Projection
                # Rotate gradient by 51.827 degrees in phase space
                cos_g = math.cos(self.giza_slope_rad)
                sin_g = math.sin(self.giza_slope_rad)
                grad_projected = grad * cos_g + torch.roll(grad, 1, dims=-1) * sin_g

                # Apply 0, 3, 6, 9 Chaotic Void Filtration
                # We logically drop gradients modifying coordinates aligned with 0, 3, 6, 9 mod 9
                flat_grad = grad_projected.view(-1)
                indices = torch.arange(flat_grad.size(0), device=grad.device)
                mod_9 = indices % 9

                # Create mask for stable nodes
                mask = torch.ones_like(flat_grad)
                mask[(mod_9 == 0) | (mod_9 == 3) | (mod_9 == 6) | (mod_9 == 9)] = 0.0

                filtered_grad = (flat_grad * mask).view_as(grad)

                # Momentum is strictly geometric Phi
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["phi_buf"] = torch.zeros_like(p)

                phi_buf = state["phi_buf"]
                phi_buf.mul_(phi_momentum).add_(filtered_grad, alpha=1 - phi_momentum)

                # Update weights
                p.add_(phi_buf, alpha=-lr)

        return loss
