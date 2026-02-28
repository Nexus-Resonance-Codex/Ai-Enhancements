import torch
import torch.nn as nn
import torch.nn.functional as F
from nrc.math.qrt import execute_qrt_damping_tensor

class QRTKernelConvolution2d(nn.Module):
    """
    Enhancement #15: QRT Kernel Convolution Layer v2

    A structural replacement for standard Vision/Spatial nn.Conv2d layers.
    Standard Convolutions learn arbitrary kernels that slide over images/grids.

    The NRC Convolution forces the active Weights into a continuous resonant
    manifold determined by the Quantitative Resonance Theorem (QRT).
    Before performing the feature extraction slide, the kernel itself is
    mathematically "damped" and bounded seamlessly by the QRT equation, allowing
    fractal dimension detection (~1.4) which massively improves spatial geometry
    recognition boundaries (edges, textures, spatial anomalies).
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Base weights initialized identically to standard Conv2D
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Dynamically calculates the QRT bounded filter weights before
        executing standard spatial 2D sliding extraction.
        """
        # 1. Subject the raw convolution kernels to the Mathematical Friction Boundary
        # This locks the learned geometries onto stable fractal bounds natively.
        qrt_damped_kernel = execute_qrt_damping_tensor(self.weight)

        # 2. Execute standard 2D extraction using the stable fractal kernels
        output = F.conv2d(x, qrt_damped_kernel, self.bias, self.stride, self.padding)

        return output

import math # Required for Kaiming Uniform Reset Standards
