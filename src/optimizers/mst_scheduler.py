import math
from torch.optim.lr_scheduler import _LRScheduler

class MSTScheduler(_LRScheduler):
    """
    Modular Synchronisation Theory Scheduler.
    
    Cycles learning rate based on the chaotic Lyapunov exponent lambda = 0.381
    and the Golden Ratio modular step bounds, heavily anchored on the period of 7.
    """
    def __init__(self, optimizer, base_lr=1e-3, phi=1.6180339887, last_epoch=-1):
        self.phi = phi
        self.base_lr = base_lr
        self.mst_lambda = 0.381
        super(MSTScheduler, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        # MST Phase calculation:
        # LR pulses according to a combination of the Pisano period and the 7-adic anchor
        step = self.last_epoch + 1
        
        # 24 is the Pisano period of Fibonacci mod 9
        pisano_phase = (step % 24) / 24.0
        
        # 7 is the stabilizing anchor
        anchor_phase = (step % 7) / 7.0
        
        # Calculate chaotic resonance envelope
        envelope = math.exp(-self.mst_lambda * (step / 1000.0))
        
        # Combine phases using Golden Ratio
        pulse = math.sin(pisano_phase * math.pi * 2) * self.phi + math.cos(anchor_phase * math.pi * 2) * (1.0 / self.phi)
        
        # Normalize and apply envelope
        multiplier = abs(pulse) * envelope
        
        return [base_lr * multiplier for base_lr in self.base_lrs]
