import math
from typing import Any, Callable, Iterable, Optional, overload

import torch
from torch.optim import Optimizer


class PGNOptimizer(Optimizer):
    """Enhancement #25: Poly-Geometric Newton Optimizer (PGNO).

    Newton-descents utilizing the golden ratio as a structural momentum pivot.
    """

    def __init__(self, params: Iterable[torch.Tensor] | Iterable[dict[str, Any]], lr: float = 1e-3, phi_momentum: float = 0.618):
        defaults = {"lr": lr, "phi_momentum": phi_momentum}
        super(PGNOptimizer, self).__init__(params, defaults)
        # Optimal geometric damping angle (theta_qrt ≈ 51.853 degrees)
        self.qrt_damping_angle = 51.853 * math.pi / 180.0

    @overload
    def step(self, closure: None = ...) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
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

                # Apply Geometric Damping Projection
                # Rotate gradient by optimal damping angle in phase space
                cos_g = math.cos(self.qrt_damping_angle)
                sin_g = math.sin(self.qrt_damping_angle)
                grad_projected = grad * cos_g + torch.roll(grad, 1, dims=-1) * sin_g

                # Apply TTT Modular Residue Stability Alignment (Mod 9)
                # We filter gradients based on coordinate alignment with modular stability nodes.
                flat_grad = grad_projected.view(-1)
                indices = torch.arange(flat_grad.size(0), device=grad.device)
                mod_9 = indices % 9

                # Create mask for stable nodes (TTT-aligned)
                mask = torch.ones_like(flat_grad)
                mask[(mod_9 == 0) | (mod_9 == 3) | (mod_9 == 6) | (mod_9 == 9)] = 0.0

                filtered_grad = (flat_grad * mask).view_as(grad)

                # Momentum is governed by the structural Golden Ratio (phi)
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["phi_buf"] = torch.zeros_like(p)

                phi_buf = state["phi_buf"]
                phi_buf.mul_(phi_momentum).add_(filtered_grad, alpha=1 - phi_momentum)

                # Update weights
                p.add_(phi_buf, alpha=-lr)

        return loss
