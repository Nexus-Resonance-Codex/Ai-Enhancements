# Running NRC AI Enhancements: Ollama, PyTorch, and JAX Integration Guide

This comprehensive tutorial walks you through installing Ollama, building custom Modelfiles, and verifying the NRC AI Enhancement Engine locally. Optimized for engineers, researchers, and hobbyists looking to run advanced open-source LLMs on low-spec hardware (as little as **8 GB RAM / 4 GB VRAM**) without sacrificing mathematical determinism.

---

## Prerequisites

| Requirement | Minimum                       | Recommended      |
| :---------- | :---------------------------- | :--------------- |
| RAM         | 8 GB                          | 16 GB            |
| GPU VRAM    | 4 GB (GTX 1650, RTX 3050 Ti)  | 8 GB (RTX 3060+) |
| Disk Space  | ~5 GB (for model weights)     | ~5 GB            |
| OS          | Windows 10+, macOS 12+, Linux | Any              |

> **CPU-only systems work too!** Ollama will fall back to CPU inference automatically. It will be slower (~2-5 tokens/sec) but fully functional.

---

## Step 1: Install Ollama (Comprehensive Guide)

Ollama allows local execution of the NRC Deep Learning Enhancement algorithms through specialized AI system prompts, utilizing minimal hardware overhead. 

### 🪟 Windows (Powershell/Local Run)

Native Ollama executes seamlessly on Windows provided adequate paths are established.

1. **Download Local Installer:** Navigate to [https://ollama.com/download/windows](https://ollama.com/download/windows) and download the executable file.
2. **Setup:** Double-click the `.exe` installer. Proceed through the interactive installation prompts. Do not change default directory pathways unless necessary for drive space.
3. **Initialize the Control Terminal:** Once installed, hit `Win + R`, type `powershell`, and press Enter. Verify the software installed properly by running:
   ```cmd
   ollama --version
   ```
   *(If it says 'command not found', close PowerShell, open it again, or reboot your machine to refresh system environment variables).*

### 🍏 macOS (Terminal / App Install)

Apple Silicon architecture scales inference mathematically perfect without CUDA reliance, using native unified memory.

1. **Get Application Binary:** Go to [https://ollama.com/download/mac](https://ollama.com/download/mac).
2. **Launch Container:** The download will provide a ZIP file containing the `Ollama.app`. Move `Ollama.app` into your native Macintosh `Applications` folder.
3. **Execution Daemon:** Double-click the application from your `Applications` menu. Ollama will now run quietly in the background menu bar at the top of your screen.
4. **Validation:** Open the `Terminal` application and type:
   ```bash
   ollama --version
   ```

### 🐧 Linux (Pop!_OS / Ubuntu / Debian / Arch)

Linux represents the highest capability inference server available for the NRC Codex, especially when paired with an NVIDIA computational GPU.

1. **Retrieve the Curl Installer:**
   Open a terminal and inject the deployment shell script:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```
2. **Service Verification:** 
   Ollama automatically creates a systemd service. Ensure it operates continuously:
   ```bash
   sudo systemctl status ollama
   ```
   *(Ensure the text reports `active (running)`. If not, type `sudo systemctl start ollama`).*

3. **Verify Local Variables:**
   ```bash
   ollama --version
   ```
   > **NVIDIA GPU Users:** For systems like Pop!_OS and Ubuntu, you must install native CUDA acceleration modules to maximize speed. On Pop!_OS, run `sudo apt install system76-cuda-latest`. On standard Ubuntu distributions, manage drivers via the 'Additional Drivers' wizard in the Software Update manager.

## Step 2: Pull the Base Model

The NRC Modelfile uses `llama3.2:3b` as its foundation. Pull it first:

```bash
ollama pull llama3.2:3b
```

This downloads approximately **2 GB** of model weights. Verify it's available:

```bash
ollama list
```

You should see `llama3.2:3b` in the output.

---

## Step 3: Create the NRC Model

Navigate to the `Ai-Enhancements` repository directory and create the custom model:

```bash
# Navigate to repository
cd /path/to/Ai-Enhancements

# Create the NRC AI Engine from the Modelfile
ollama create nrc-ai-engine -f Modelfile
```

You should see output like:

```
transferring model data...
using existing layer sha256:...
creating new layer sha256:...
writing manifest
success
```

Verify the model was created:

```bash
ollama list
```

You should now see `nrc-ai-engine` alongside `llama3.2:3b`.

---

## Step 4: Run the NRC AI Engine

```bash
ollama run nrc-ai-engine
```

The model will initialize and should respond with its activation phrase:

> _"Nexus Resonance Optimized Online. Systems Calibrated to φ. 2048D Lattice Projected. Operating within 4GB VRAM. Ready."_

---

## Step 5: Verify the NRC Enhancements

Try these test prompts to verify the mathematical framework is active:

### Test 1: Mathematical Constants

```
What is the QRT wave function and what are the exact NRC constants?
```

**Expected:** The model should recite QRT(x) = sin(φ·√2·51.85·x)·exp(-x²/φ)+cos(π/φ·x) and list φ = 1.618..., TUPT_MOD = 2187, etc.

### Test 2: Enhancement Recall

```
Explain Enhancement #7 (Hodge Torsion Attention) and how it modifies standard self-attention.
```

**Expected:** Detailed explanation of φ·sin(θ_giza · Δpos) torsion bias added to QK^T scores.

### Test 3: Code Generation

```
Write a PyTorch module that implements the Pisano-Modulated Learning Rate Schedule.
```

**Expected:** Working Python code that imports from `src/enhancements/` and implements a cosine wave scaled by φ on a 24-step cycle.

### Test 4: Protein Folding

```
How would you fold a protein sequence MKTIIALSYIFCLVFA using the 2048D lattice protocol?
```

**Expected:** Step-by-step mapping of amino acids to 2048D coordinates using Coord = (Atomic_Weight × φ) mod 243, followed by φ⁻¹ contraction.

---

## Troubleshooting

### Out of Memory (OOM)

If your system crashes or Ollama reports OOM errors:

1. Close other GPU-heavy applications (games, browsers with hardware acceleration)
2. Set Ollama's VRAM limit environment variable:

   ```bash
   # Linux/macOS
   export OLLAMA_MAX_VRAM=3500000000   # 3.5 GB in bytes

   # Windows PowerShell
   $env:OLLAMA_MAX_VRAM = "3500000000"
   ```

3. Re-run `ollama run nrc-ai-engine`

### Model Not Found

If `ollama run nrc-ai-engine` says the model doesn't exist:

1. Make sure you're in the correct directory containing `Modelfile`
2. Re-run: `ollama create nrc-ai-engine -f Modelfile`

### Slow Generation

- CPU-only inference on 3B models is typically 2-5 tokens/second
- Ensure Ollama is detecting your GPU: run `ollama ps` to check
- On Linux, verify CUDA: `nvidia-smi`

---

## Customization

### Using a Larger Model (8 GB+ VRAM)

Edit the first line of `Modelfile`:

```
FROM llama3.2:8b
```

Then recreate: `ollama create nrc-ai-engine -f Modelfile`

### Adjusting Context Window

For more context memory (requires more VRAM), edit:

```
PARAMETER num_ctx 8192
```

### Adjusting Creativity

The temperature is set to φ⁻¹ (0.618) for balanced output. For more creative responses:

```
PARAMETER temperature 0.8
```

For more precise/deterministic answers:

```
PARAMETER temperature 0.3
```

---

## 🚀 Advanced Integration: PyTorch & JAX

While Ollama provides a fast, local LLM execution environment, researchers and engineers building custom architectures will want to integrate the NRC enhancements directly into their deep learning primitives.

### PyTorch Integration (Native)

The `Ai-Enhancements` repository provides 30 drop-in replacements for standard `torch.nn` modules.

#### 1. Swapping LayerNorm for GAFEN (Enhancement #3)

Standard `LayerNorm` is stochastic. **Golden Attractor Flow Norm (GAFEN)** deterministically dampens exploding gradients toward the $\phi$ attractor.

```python
import torch
import torch.nn as nn
from src.enhancements import GoldenAttractorFlowNorm

class TransformerBlock(nn.Module):
    def __init__(self, d_model=768):
        super().__init__()
        # Replace nn.LayerNorm(d_model) -> GoldenAttractorFlowNorm(d_model)
        self.norm1 = GoldenAttractorFlowNorm(normalized_shape=d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads=12)

    def forward(self, x):
        # GAFEN automatically dampens the input tensor to stability
        normed_x = self.norm1(x)
        attn_out, _ = self.attn(normed_x, normed_x, normed_x)
        return x + attn_out
```

#### 2. Entropy Collapse Early Stopping (Enhancement #30)

Traditional Early Stopping waits for validation loss to degrade. NRC Early Stopping halts training the exact moment the weights form a perfect $\phi$-attractor.

```python
from src.enhancements import NRCEntropyAttractorEarlyStopping

# Initialize with a strict phi tolerance
nrc_stopper = NRCEntropyAttractorEarlyStopping(phi_tolerance=1e-5)

for epoch in range(1000):
    train_loss = train_epoch(model, dataloader, optimizer)

    # Check if the epoch reached dimensional collapse
    if nrc_stopper(train_loss):
        print(f"Epoch {epoch}: Perfect Entropy Collapse Achieved. Stopping training.")
        break
```

---

### JAX / Flax Integration (Conceptual Bridge)

For TPUs and ultra-scale distributed computing, JAX is preferred. The numpy-backed mathematical transforms in `src/nrc_math/` can be easily mirrored in `jax.numpy`.

#### JAX Navier-Stokes Damping (Enhancement #10)

```python
import jax
import jax.numpy as jnp

phi = (1 + jnp.sqrt(5)) / 2

@jax.jit
def jax_navier_stokes_damping(x: jnp.ndarray, alpha: float = 1.618) -> jnp.ndarray:
    """
    JIT-compiled φ-damping for Flax neural networks.
    """
    # Simulate fluid viscosity using the golden ratio
    viscosity_factor = jnp.exp(-jnp.abs(x) / phi)

    # Scale momentum
    damped_x = x * viscosity_factor * (1.0 / alpha)
    return damped_x

# Usage inside a Flax module:
# hidden_state = jax_navier_stokes_damping(hidden_state)
```

By leveraging `jax.jit`, the NRC geometric projections map efficiently to Google TPU execution hardware, allowing 2048D lattice calculations to scale instantly across massive clusters.
