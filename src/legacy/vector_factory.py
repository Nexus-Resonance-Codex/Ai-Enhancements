"""NEXUS RESONANCE CODEX - Phase 4: Multi-Manifold Synthetic Intelligence (MSI).

Module: vector_factory.py
Protocol: NRC Vector Protocol (NVP) V1.0.7.
"""

import json
from typing import Any, Dict


class NRCVectorFactory:
    """Standardizes cross-manifold data into the NVP embedding manifold.

    Mapping Strategy:
    - Phase 1 (Protein): RMSD/pLDDT -> Semantic Coordinate
    - Phase 2 (Meta): Absorption/S-Param -> Resonant Coordinate
    - Phase 3 (Quantum): T2/P_ex -> Coherence Coordinate
    """

    def __init__(self, phi: float = 1.61803398875):
        self.phi = phi
        self.version = "1.0.7"

    def generate_nvp_metadata(self, source_manifold: str, data: Dict[str, Any]) -> str:
        """Generates a unified NVP metadata artifact for cross-manifold RAG."""
        # Calculate Resonant Embedding Index (REI)
        # REI = (Value_Manifold / Phi^2) mod TTT-7
        base_val = sum([float(v) for v in data.values() if isinstance(v, (int, float))])
        rei = (base_val / (self.phi**2)) % 7

        metadata = {
            "nrc_header": {
                "manifold": source_manifold,
                "protocol": f"NVP-{self.version}",
                "ttt7_stability": "STABLE" if self._is_ttt7(rei) else "CHAOTIC",
            },
            "embedding": {"rei": round(rei, 7), "phi_spiral_step": int(base_val * self.phi)},
            "source_data": data,
        }

        return json.dumps(metadata, indent=2)

    def _is_ttt7(self, val: float) -> bool:
        scaled = int(val * 1e7)
        if scaled == 0:
            return True
        dr = scaled % 9
        return (dr if dr != 0 else 9) in {1, 2, 4, 5, 7, 8}


if __name__ == "__main__":
    factory = NRCVectorFactory()

    # Test: Phase 1 Protein Data
    p1_data = {"rmsd": 0.12, "plddt": 92.5}
    # Test: Phase 2 Metamaterial Data
    p2_data = {"absorption": 0.91, "stability": 0.215567}

    print("--- NRC PHASE 4: NVP MANIFOLD VECTORIZATION ---")
    print("PHASE 1 (PROTEIN) NVP:")
    print(factory.generate_nvp_metadata("PROTEIN_FOLDING", p1_data))
    print("\nPHASE 2 (META) NVP:")
    print(factory.generate_nvp_metadata("META_MATERIALS", p2_data))
    print("-----------------------------------------------")
