import torch
import math

PHI = (1.0 + math.sqrt(5.0)) / 2.0
GIZA_SLOPE = 51.85 # Specified explicitly
SQRT_2 = math.sqrt(2.0)
PI = math.pi

def qrt_damping(x: torch.Tensor) -> torch.Tensor:
    """
    Quantum Resonance Theory (QRT) wave function.
    QRT(x) = sin(φ * sqrt(2) * 51.85 * x) * exp(-x**2 / φ) + cos(π / φ * x)

    Acts as a damping function with a fractal dimensionality of ~1.4.
    """
    freq_sin = PHI * SQRT_2 * GIZA_SLOPE
    freq_cos = PI / PHI

    term1 = torch.sin(freq_sin * x)
    term2 = torch.exp(-(x**2) / PHI)
    term3 = torch.cos(freq_cos * x)

    return (term1 * term2) + term3


def execute_qrt_damping_tensor(x: torch.Tensor) -> torch.Tensor:
    """
    Alias for qrt_damping() — used by Enhancement #10 (Navier-Stokes Damping
    Regulariser) and other modules that call the QRT wave function on full
    hidden-state tensors.

    QRT(x) = sin(φ · √2 · 51.85 · x) · exp(-x² / φ) + cos(π/φ · x)
    """
    return qrt_damping(x)
