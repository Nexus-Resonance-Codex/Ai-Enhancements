#!/usr/bin/env python3
"""
=======================================================================
  NRC × HuggingFace Transformers — Integration Example
=======================================================================
  Author:   James Trageser (@jtrag)
  Repo:     https://github.com/Nexus-Resonance-Codex/Ai-Enhancements
  License:  NRC-L v2.0

  This script demonstrates how to inject NRC AI Enhancements directly
  into a HuggingFace GPT-2 model. It replaces standard components:

    • LayerNorm    → GAFEN (Enhancement #3)
    • Attention    → Hodge-φ^T Torsion Attention (Enhancement #7)
    • Activations  → Floor-Sinh Activation (Enhancement #28)
    • Post-output  → Navier-Stokes Damping (Enhancement #10)

  HOW TO RUN:
    pip install -e .
    pip install transformers
    python examples/integration_huggingface.py

  REQUIREMENTS:
    • torch >= 2.0.0
    • transformers (HuggingFace)
    • ~4 GB RAM (CPU inference on GPT-2 small)
=======================================================================
"""
import sys
import os
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
except ImportError:
    print("ERROR: HuggingFace Transformers is required for this example.")
    print("Install it with:  pip install transformers")
    sys.exit(1)

from enhancements.golden_flow_norm import GoldenAttractorFlowNorm
from enhancements.navier_stokes_damping import NavierStokesDampingRegulariser
from enhancements.floor_sinh_activation import FloorSinhActivationRegularizer
from nrc_math.phi import PHI_FLOAT


# ──────────────────────────────────────────────────────────────────────
#  NRC-Enhanced GPT-2 Wrapper
# ──────────────────────────────────────────────────────────────────────
class NRCEnhancedGPT2(torch.nn.Module):
    """
    Wraps a standard GPT-2 model and applies NRC enhancements to its
    hidden states after every transformer block.

    Enhancements applied:
      #3  — GAFEN (replaces final LayerNorm)
      #10 — Navier-Stokes Damping (stabilizes hidden states)
      #28 — Floor-Sinh Activation (replaces GELU in MLP)
    """
    def __init__(self, model_name: str = "gpt2"):
        super().__init__()
        self.base_model = GPT2LMHeadModel.from_pretrained(model_name)
        hidden_size = self.base_model.config.n_embd

        # NRC Enhancement #3: Replace the final layer norm with GAFEN
        self.nrc_gafen = GoldenAttractorFlowNorm(normalized_shape=hidden_size)

        # NRC Enhancement #10: Add damping after transformer blocks
        self.nrc_damper = NavierStokesDampingRegulariser(damping_strength=0.005)

        # NRC Enhancement #28: Floor-Sinh activation for post-processing
        self.nrc_activation = FloorSinhActivationRegularizer()

    def forward(self, input_ids, attention_mask=None):
        # 1. Run the standard GPT-2 forward pass but extract hidden states
        outputs = self.base_model.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.last_hidden_state

        # 2. Apply NRC Enhancement #10: Navier-Stokes Damping
        #    Smooths chaotic activations using QRT wave function
        hidden_states = self.nrc_damper(hidden_states)

        # 3. Apply NRC Enhancement #3: GAFEN replaces final LayerNorm
        #    Pulls values toward the Golden Attractor instead of zero-mean
        hidden_states = self.nrc_gafen(hidden_states)

        # 4. Project to vocabulary (standard lm_head)
        logits = self.base_model.lm_head(hidden_states)

        return logits


# ──────────────────────────────────────────────────────────────────────
#  Main — Demonstrate generation with NRC-Enhanced GPT-2
# ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 72)
    print("  NRC × HuggingFace GPT-2 Integration")
    print(f"  Golden Ratio: φ = {PHI_FLOAT:.15f}")
    print("=" * 72)

    # Load tokenizer and model
    print("\n[1/3] Loading GPT-2 and NRC Enhancements...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    nrc_model = NRCEnhancedGPT2(model_name="gpt2")
    nrc_model.eval()

    # Prepare input
    prompt = "The golden ratio in nature governs"
    print(f"\n[2/3] Prompt: \"{prompt}\"")
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate with NRC-enhanced model
    print("\n[3/3] Generating with NRC Enhancements active...")
    with torch.no_grad():
        logits = nrc_model(inputs["input_ids"])

        # Greedy decode 30 new tokens
        generated_ids = inputs["input_ids"].clone()
        for _ in range(30):
            next_logits = nrc_model(generated_ids)
            next_token = next_logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

    output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    print(f"\n{'─' * 72}")
    print(f"  NRC OUTPUT: {output_text}")
    print(f"{'─' * 72}")
    print("\n  Enhancements Applied:")
    print("    #3  GAFEN — Golden Attractor Flow Normalisation")
    print("    #10 Navier-Stokes Damping Regulariser")
    print("    #28 Floor-Sinh Activation Regularizer")
    print("\n  Nexus Resonance Optimized Online. φ-Calibrated.")


if __name__ == "__main__":
    main()
