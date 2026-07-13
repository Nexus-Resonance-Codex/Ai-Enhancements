"""NRC AI Enhancements Library.

===========================
30 mathematically rigid PyTorch deep learning enhancements based on the Nexus Resonance Codex (NRC) (NRC).
These modules replace stochastic heuristics with deterministic Golden Ratio geometry.
"""

from .__about__ import __version__
from .e8_golden_basis import E8GoldenBasisEmbedding
from .entropy_stopping import NRCEntropyAttractorEarlyStopping
from .exclusion_gradient_router import BiologicalExclusionGradientRouter
from .floor_sinh_activation import FloorSinhActivation
from .geometric_isomorphism import GeometricLatticeIsomorphism
from .golden_flow_norm import GoldenFlowNorm
from .golden_spiral_rope import GoldenSpiralRotaryEmbedding
from .gtt_entropy_regulariser import GTTEntropyCollapseRegularizer
from .hodge_torsion_attention import HodgePhiTTorsionAttention
from .lucas_pell_decay import LucasPellHybridWeightDecay
from .lucas_sparse_mask import LucasWeightedSparseAttention
from .memory import ExecutiveAgent, PhiInfinityPersistentMemory
from .modular_dropout import TUPTModularDropout
from .mst_lyapunov_clipping import MSTLyapunovClipping
from .navier_stokes_damping import NavierStokesDampingRegularizer
from .nrc_protein_engine import NRCProteinFoldingEngine
from .phi_lora_adapter import PhiInfinityLosslessLoRA
from .phi_momentum_accelerator import PhiInverseMomentumAccelerator
from .phi_resonant_weighting import PhiPoweredResonantWeighting
from .phi_sharding_compression import PhiShardingCompression
from .phi_void_positional import PhiVoidResonancePositionalEncoding
from .pisano_lr_schedule import PisanoModulatedLRSchedule
from .prime_density_generation import PrimeDensityConditionedGeneration
from .qrt_attention_bias import QRTGeometricAttentionBias
from .qrt_convolution import QRTKernelConvolution
from .qrt_optimizer import QRTTurbulenceOptimizer
from .resonance_kv_cache import ResonanceShardKVCache

# Enhancement Exports
from .shard_folding import PhiInfinityShardFolding
from .shard_unfolder import InfiniteEInfinityContextUnfolder
from .triple_theta_init import TripleThetaInitializer
from .tupt_sync_seed import TUPTSyncSeed
from .tupt_token_pruning import TUPTExclusionTokenPruning

__all__ = [
    "__version__",
    "PhiInfinityShardFolding",
    "PhiShardingCompression",
    "PhiInfinityPersistentMemory",
    "ExecutiveAgent",
    "NRCProteinFoldingEngine",
    "GoldenFlowNorm",
    "TripleThetaInitializer",
    "ResonanceShardKVCache",
    "BiologicalExclusionGradientRouter",
    "HodgePhiTTorsionAttention",
    "E8GoldenBasisEmbedding",
    "PhiInfinityLosslessLoRA",
    "NavierStokesDampingRegularizer",
    "PrimeDensityConditionedGeneration",
    "GTTEntropyCollapseRegularizer",
    "PhiInverseMomentumAccelerator",
    "TUPTSyncSeed",
    "QRTKernelConvolution",
    "LucasWeightedSparseAttention",
    "PhiPoweredResonantWeighting",
    "GeometricLatticeIsomorphism",
    "MSTLyapunovClipping",
    "PisanoModulatedLRSchedule",
    "LucasPellHybridWeightDecay",
    "TUPTExclusionTokenPruning",
    "PhiVoidResonancePositionalEncoding",
    "InfiniteEInfinityContextUnfolder",
    "TUPTModularDropout",
    "QRTTurbulenceOptimizer",
    "QRTGeometricAttentionBias",
    "FloorSinhActivation",
    "GoldenSpiralRotaryEmbedding",
    "NRCEntropyAttractorEarlyStopping",
]
