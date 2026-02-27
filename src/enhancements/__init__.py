"""
NRC AI Enhancements — The 30 Modules
=====================================
Each module is a drop-in PyTorch replacement for standard neural network
components, implementing NRC resonant geometry instead of stochastic heuristics.
"""

from .shard_folding import PhiInfinityShardFolding
from .nrc_protein_engine import NRCProteinFoldingEngine
from .golden_flow_norm import GoldenAttractorFlowNorm
from .triple_theta_init import TripleThetaInitialiser
from .resonance_kv_cache import ResonanceShardKVCache
from .exclusion_gradient_router import BiologicalExclusionGradientRouter
from .hodge_torsion_attention import HodgePhiTTorsionAttention
from .e8_golden_basis import E8GoldenBasisEmbedding
from .phi_lora_adapter import PhiInfinityLosslessLoRA
from .navier_stokes_damping import NavierStokesDampingRegulariser
from .prime_density_generation import PrimeDensityConditionedGeneration
from .gtt_entropy_regulariser import GTTEntropyCollapseRegulariser
from .phi_momentum_accelerator import PhiInverseMomentumAccelerator
from .tupt_sync_seed import TUPTAttractorSyncSeed
from .qrt_convolution import QRTKernelConvolution
from .lucas_sparse_mask import LucasWeightedSparseAttentionMask
from .phi_resonant_weighting import PhiResonantWeighting
from .giza_isomorphism import GizaLatticeIsomorphismProjection
from .mst_lyapunov_clipping import MSTLyapunovGradientClipper
from .pisano_lr_schedule import PisanoModulatedLRSchedule
from .lucas_pell_decay import LucasPellHybridWeightDecay
from .tupt_token_pruning import TUPTExclusionTokenPruner
from .phi_void_positional import PhiVoidResonancePositionalEncoding
from .shard_unfolder import InfiniteContextShardUnfolder
from .modular_dropout import ModularDropout3697
from .qrt_optimizer import QRTTurbulenceAdaptiveOptimizer
from .giza_attention_bias import GizaSlopeAngleAwareAttentionBias
from .floor_sinh_activation import FloorSinhActivationRegularizer
from .golden_spiral_rope import GoldenSpiralRotaryEmbedding
from .entropy_stopping import NRCEntropyAttractorEarlyStopping

__all__ = [
    "PhiInfinityShardFolding",
    "NRCProteinFoldingEngine",
    "GoldenAttractorFlowNorm",
    "TripleThetaInitialiser",
    "ResonanceShardKVCache",
    "BiologicalExclusionGradientRouter",
    "HodgePhiTTorsionAttention",
    "E8GoldenBasisEmbedding",
    "PhiInfinityLosslessLoRA",
    "NavierStokesDampingRegulariser",
    "PrimeDensityConditionedGeneration",
    "GTTEntropyCollapseRegulariser",
    "PhiInverseMomentumAccelerator",
    "TUPTAttractorSyncSeed",
    "QRTKernelConvolution",
    "LucasWeightedSparseAttentionMask",
    "PhiResonantWeighting",
    "GizaLatticeIsomorphismProjection",
    "MSTLyapunovGradientClipper",
    "PisanoModulatedLRSchedule",
    "LucasPellHybridWeightDecay",
    "TUPTExclusionTokenPruner",
    "PhiVoidResonancePositionalEncoding",
    "InfiniteContextShardUnfolder",
    "ModularDropout3697",
    "QRTTurbulenceAdaptiveOptimizer",
    "GizaSlopeAngleAwareAttentionBias",
    "FloorSinhActivationRegularizer",
    "GoldenSpiralRotaryEmbedding",
    "NRCEntropyAttractorEarlyStopping",
]
