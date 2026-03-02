from typing import Iterable

import torch
from nrc.math.mst import mst_step


class MSTLyapunovGradientClipping:
    """
    Enhancement #19: MST-Lyapunov Gradient Clipping Stabilizer

    Standard deep learning handles exploding gradients via a rigid structural
    cutoff (e.g. max_norm = 1.0).

    The NRC maps gradient noise onto Lyapunov Exponents determining chaotic
    divergence bounds. Instead of a hard literal cutoff which damages topological
    geometries, this module iterates through gradient pools and dynamically scales
    the magnitude of explosive spikes using the continuous Macro-Scale Theorem (MST)
    decay function. High Lyapunov noise is compressed linearly via continuous calculus
    rather than completely sheared off.
    """
    @staticmethod
    def clip_grad_mst_norm_(parameters: Iterable[torch.Tensor], max_lyapunov_threshold: float = 2.0) -> torch.Tensor:
        """
        Calculates gradient scale norms, and functionally subjects tensors
        breaching Lyapunov divergence markers to MST continuous friction.
        """
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]

        parameters = [p for p in parameters if p.grad is not None]
        if len(parameters) == 0:
            return torch.tensor(0.)

        # Calculate standard global norm measuring entire gradient explosion field
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters]), 2)

        # Check if the global explosion breaches the Lyapunov safety structural bound
        if total_norm > max_lyapunov_threshold:
            for p in parameters:
                # 1. Identify specific gradients exceeding the Lyapunov bound individually
                spike_mask = torch.abs(p.grad) > max_lyapunov_threshold

                # 2. Extract strictly the explosive magnitudes
                spikes = p.grad[spike_mask]

                # 3. Subject the raw gradient explosions explicitly to the continuous MST bounding decay
                damped_spikes = mst_step(spikes)

                # 4. Integrate the mathematically decayed safe gradients back into the main autograd pipeline
                p.grad[spike_mask] = damped_spikes

        return total_norm
