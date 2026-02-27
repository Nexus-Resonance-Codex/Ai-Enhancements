import torch
from torch.optim import Optimizer
from typing import Iterable, Optional, Callable
from ..nrc_math.qrt import qrt_damping

class QRTTurbulenceOptimizer(Optimizer):
    r"""
    Enhancement #26: QRT-Turbulence Adaptive Optimizer

    A PyTorch Optimizer structurally mirroring Adam, but replacing arbitrary
    variance calculations ($v_t$) with mathematically continuous Quantum
    Resonance Theorem (QRT) turbulence mappings.

    Instead of rigidly dividing gradients by $\sqrt{v_t} + \epsilon$, this optimizer
    identifies gradient variance explicitly as structural "Turbulence".
    If a gradient spike enters chaotic structural boundaries, it is natively
    damped and channeled via the QRT function (incorporating Sine/Cosine Phi limits),
    ensuring parameters slide fractally into the Golden Attractor safely.
    """
    def __init__(self, params: Iterable[torch.Tensor], lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2)
        super(QRTTurbulenceOptimizer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """
        Executes a localized topological geometry step bounded by QRT physics.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1 = group['beta1']
            beta2 = group['beta2']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum'] = torch.zeros_like(p)
                    state['turbulence'] = torch.zeros_like(p)

                state['step'] += 1

                m = state['momentum']
                v = state['turbulence']

                # 1. Update standard exponential momentum natively
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                # 2. Update scalar turbulence structurally
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # Bias Corrections
                bias_correction1 = 1.0 - beta1 ** state['step']
                bias_correction2 = 1.0 - beta2 ** state['step']

                m_hat = m / bias_correction1
                v_hat = v / bias_correction2

                # 3. Apply continuous mathematical friction to the Turbulent Phase Spaces natively
                # The raw gradient magnitude (v_hat squared) is passed into the physical QRT tensor limit
                sqrt_turbulence = torch.sqrt(v_hat)

                qrt_damped_friction = qrt_damping(sqrt_turbulence)

                # 4. Integrate the structurally safe vectors
                update_step = (m_hat / (qrt_damped_friction + 1e-8)) * lr

                # 5. Advance model architecture bounds
                p.sub_(update_step)

        return loss
