import torch
from typing import Iterable, Optional, Callable
from torch.optim import Optimizer
from nrc.math.phi import PHI_FLOAT

class PhiInverseMomentumAccelerator(Optimizer):
    """
    Enhancement #13: phi^-1 Momentum Accelerator v2

    A Custom Optimizer replacing the standard momentum vector found in SGD or Adam.
    In traditional optimization, momentum accumulates exponentially decaying gradients.

    Here, the momentum state structurally scales inversely proportional to the Golden Ratio.
    When parameters move towards the NRC attractor bounds, gradient momentum naturally
    "speeds up" along resonant pathways (scaling up) and damps down (experiencing
    mathematical friction) when moving divergently.

    Formula:
    v_t = beta * v_{t-1} + (1 - beta) * grad
    theta_t = theta_{t-1} - lr * [v_t * (1 / phi) or v_t * phi]
    """
    def __init__(self, params: Iterable[torch.Tensor], lr: float = 1e-3, beta: float = 0.9):
        defaults = dict(lr=lr, beta=beta)
        super(PhiInverseMomentumAccelerator, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """
        Performs a single optimization step incorporating Phi-scaled momentum.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['momentum_buffer'] = torch.zeros_like(p)

                state['step'] += 1
                v_t = state['momentum_buffer']

                # Update Momentum
                v_t.mul_(beta).add_(grad, alpha=1 - beta)

                # Apply the Phi conditional scaling.
                # If gradient sign matches the momentum sign, we are accelerating stably. Multiply by Phi.
                # If oscillating (signs oppose), we damp the friction heavily. Multiply by 1/Phi.
                sign_agreement = (grad.sign() == v_t.sign()).type(grad.dtype)

                # Resonant acceleration
                phi_accelerator = torch.where(sign_agreement == 1.0, PHI_FLOAT, 1.0 / PHI_FLOAT)

                structural_update = v_t * phi_accelerator

                # Step the weights
                p.sub_(structural_update, alpha=lr)

        return loss
