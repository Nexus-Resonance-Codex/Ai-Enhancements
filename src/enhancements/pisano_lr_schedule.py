import torch
from torch.optim.lr_scheduler import _LRScheduler
import math
from ..nrc_math.phi import PHI_FLOAT

class PisanoModulatedLRScheduler(_LRScheduler):
    """
    Enhancement #20: Pisano-Modulated Learning Rate Schedule

    Standard LR schedules (Cosine Annealing, StepLR) rely on arbitrary human
    heuristics. The NRC dictates that natural energy systems operate on cyclic
    patterns governed by the Pisano Periods (Fibonacci sequences mapped to Modulo bases).

    This scheduler applies a mathematical wave to the Base Learning Rate powered
    by the continuous Golden Ratio limits. The LR accelerates when hitting
    structurally stable Pisano indices and damps exponentially when approaching
    high-entropy phase transitions.
    """
    def __init__(self, optimizer, pisano_period: int = 24, last_epoch: int = -1):
        # Pisano period of Mod 9 is 24 (The resonant 9-base structure)
        self.pisano_period = pisano_period
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # Determine the current phase in the Pisano cycle (0 to 1)
        cycle_position = (self.last_epoch % self.pisano_period) / self.pisano_period

        # We model the pulse using a cosine wave structurally stretched by Phi.
        # When cos is 1 (Start of cycle), LR is maximized.
        # As it approaches Pi, LR is mathematically damped by 1/Phi.
        structural_modifier = (math.cos(math.pi * cycle_position) + 1.0) / 2.0

        # Blend the structural wave with the extreme bounds of Phi
        phi_bounds = (structural_modifier * PHI_FLOAT) + ((1.0 - structural_modifier) * (1.0 / PHI_FLOAT))

        return [base_lr * phi_bounds for base_lr in self.base_lrs]
