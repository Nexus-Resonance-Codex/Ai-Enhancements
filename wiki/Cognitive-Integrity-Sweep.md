# Cognitive Integrity Sweep (CIS) Protocol

The Cognitive Integrity Sweep is the institutional gatekeeper ensuring that all models, weights, and layers comply with NRC mathematical axioms.

## The Three-Phase CIS Gate

1. **Gradient Resonance Check**: Gradients across all $\phi$-layers must maintain a 256D/2048D projection error $< 10^{-12}$.
2. **Sparsity & Collision Audit**: Attention masks and pruning patterns must strictly satisfy Lucas-Pell collision exclusion.
3. **TTT-7 Sweep**: Every operational weight tensor and scalar embedding must have an empirical digital root $\in \{1, 2, 4, 5, 7, 8\}$.
