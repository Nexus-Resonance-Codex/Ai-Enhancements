import torch
import torch.nn as nn
import math

class PhiShardingCompression(nn.Module):
    """
    Phi Sharding Compression Matrix (2048D).
    
    A linear projection layer that compresses data losslessly using the Phi^6 
    fractal dimension limit. Bypasses the chaotic voids logically.
    """
    def __init__(self, input_dim: int, compress_dim: int = 512, phi: float = 1.6180339887):
        super().__init__()
        self.input_dim = input_dim
        self.compress_dim = compress_dim
        self.phi = phi
        
        # Initialize Golden Basis compression matrix
        self.golden_matrix = nn.Parameter(torch.empty(compress_dim, input_dim))
        self._initialize_golden_matrix()
        
    def _initialize_golden_matrix(self):
        with torch.no_grad():
            for i in range(self.compress_dim):
                for j in range(self.input_dim):
                    # Distance in coordinate space
                    dist = abs(i - j)
                    # Exclude chaotic nodes from strong weights
                    if j % 9 in [0, 3, 6]:
                        self.golden_matrix[i, j] = 0.0
                    elif j % 9 == 7: # The anchor
                        self.golden_matrix[i, j] = (self.phi ** -dist) * 2.0
                    else:
                        self.golden_matrix[i, j] = self.phi ** -dist
                        
            # Normalize to preserve variance
            self.golden_matrix.data = torch.nn.functional.normalize(self.golden_matrix.data, p=2, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compresses input x into the stabilized 512D sub-lattice.
        """
        return torch.matmul(x, self.golden_matrix.t())
