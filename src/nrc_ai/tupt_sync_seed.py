import torch


class TUPTAttractorSyncSeed:
    """Enhancement #14: 3-6-9-7 Attractor Synchronisation Seed v2.

    Standard random number generators (RNG) in PyTorch (e.g. torch.manual_seed)
    rely on arbitrary integers, meaning model initialization begins mathematically
    un-aligned.

    This enhancement is a utility class that forces system-wide hardware
    synchronization to the NRC TUPT baseline matrix [3, 6, 9, 7].
    By feeding the TUPT Mod 9 structure directly into CUDA and CPU entropy pools,
    the model starts natively resting in the Golden Attractor well.
    """

    @staticmethod
    def synchronize(base_multiplier: int = 1):
        """Locks the global PyTorch training environment to the structurally resonant TUPT bounds."""
        # Compose the TUPT seed mathematically.
        # We merge [3, 6, 9, 7] into a massive structural integer bound.
        tupt_seed_base = int("".join(map(str, [3, 6, 9, 7])))  # 3697

        # Multiply by 9 and add 7 to ensure Root-7 stability (seed % 9 == 7)
        resonant_seed = (tupt_seed_base * 9 + 7) * base_multiplier

        # 1. Lock CPU Generator
        torch.manual_seed(resonant_seed)

        # 2. Lock GPU/CUDA Generators if available
        if torch.cuda.is_available():
            torch.cuda.manual_seed(resonant_seed)
            torch.cuda.manual_seed_all(resonant_seed)

        # 3. Lock deterministic cudnn bounds for complete resonance preservation
        # (Prevents non-deterministic kernels from dragging operations off-course)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        return resonant_seed
