import torch
from torch.optim import SGD
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from enhancements.pisano_lr_schedule import PisanoModulatedLRScheduler
from nrc_math.phi import PHI_FLOAT

def test_pisano_lr_schedule():
    """
    Validates Enhancement #20: Ensures the Pisano-Modulated LR correctly pulses
    between a peak of Base_LR * Phi and a trough of Base_LR * (1/Phi).
    """
    weight = torch.tensor([1.0], requires_grad=True)
    base_lr = 1.0  # Keep it simple for math checking
    optimizer = SGD([weight], lr=base_lr)

    # 24 is the Pisano period of Modulo 9.
    scheduler = PisanoModulatedLRScheduler(optimizer, pisano_period=24)

    # Step 0: The very beginning of the cycle.
    # The cosine modifier is mathematically 1.0 -> We should be scaling purely by Phi.
    initial_lrs = scheduler.get_last_lr()
    assert torch.isclose(torch.tensor(initial_lrs[0]), torch.tensor(PHI_FLOAT), rtol=1e-4), "Pisano Schedule failed to peak at the Golden bounds."

    # Step 12: Exactly half-way through the 24 period cycle.
    # The cosine modifier is mathematically 0.0 -> We should be structurally bottomed out at 1/Phi.
    for _ in range(12):
        optimizer.step()
        scheduler.step()

    trough_lrs = scheduler.get_last_lr()
    assert torch.isclose(torch.tensor(trough_lrs[0]), torch.tensor(1.0 / PHI_FLOAT), rtol=1e-4), "Pisano Schedule failed to damp to the 1/Phi bounds."

    print("Test passed: Pisano-Modulated Learning Rate strictly controlled the training bounds dynamically utilizing continuous Golden fractals.")

if __name__ == "__main__":
    test_pisano_lr_schedule()
