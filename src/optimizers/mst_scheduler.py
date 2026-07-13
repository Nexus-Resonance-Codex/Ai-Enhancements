#  Nexus Resonance Codex (NRC) (NRC) - 2025-2026 Breakthrough Series
#  Copyright (c) 2026 James Paul Trageser (@jtrag)
#
#  Licensed under CC-BY-NC-SA-4.0 + NRC-L
#  "This work is part of the Nexus Resonance Codex (NRC) (NRC) (NRC) incorporating TTT
#  modular exclusion, phi^inf compression, 256D->729D lattice, QRT, and MST."

"""MST Scheduler: Multi-Scale Tensor Learning Rate Decay.

This module implements the MST Scheduler, which pulses the learning rate
according to structural resonance intervals defined by the Golden Ratio.
"""

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class MSTScheduler(_LRScheduler):
    """Enhancement #26: Multi-Scale Tensor (MST) Scheduler.

    Pulses learning rate according to structural resonance intervals.
    """

    def __init__(self, optimizer: Optimizer, base_lr: float = 1e-3, phi: float = 1.6180339887, last_epoch: int = -1):
        self.phi = phi
        self.base_lr = base_lr
        self.mst_lambda = 0.381
        super(MSTScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:  # type: ignore[override]
        """Calculates the resonant learning rate for each parameter group."""
        # MST Phase calculation:
        # LR pulses according to a combination of the Pisano period and the 7-adic anchor
        step = self.last_epoch + 1
        phi_phase = math.sin((math.pi / self.phi) * step) * self.mst_lambda
        anchor_7 = 1.0 if (step % 7 == 0) else 0.7

        resonant_lr = self.base_lr * (1.0 + phi_phase) * anchor_7
        return [resonant_lr for _ in self.optimizer.param_groups]
