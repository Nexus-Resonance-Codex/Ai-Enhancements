# Running NRC AI Enhancements Locally via Ollama

This guide walks you through installing, creating, running, and verifying the NRC AI Enhancement Engine on your own machine. The Modelfile is optimized for systems with as little as **8 GB RAM / 4 GB VRAM**.

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

## Step 1: Install Ollama

### Windows

1. Download the installer from [https://ollama.com/download/windows](https://ollama.com/download/windows)
2. Run the `.exe` and follow the installer prompts
3. Open **PowerShell** or **Command Prompt**
4. Verify installation:
   ```cmd
   ollama --version
   ```

### macOS

1. Download from [https://ollama.com/download/mac](https://ollama.com/download/mac)
2. Drag `Ollama.app` to your Applications folder
3. Launch Ollama from Applications
4. Open **Terminal** and verify:
   ```bash
   ollama --version
   ```

### Linux (Pop!\_OS, Ubuntu, Debian, Fedora, Arch, etc.)

```bash
# One-line install script (official)
curl -fsSL https://ollama.com/install.sh | sh

# Verify
ollama --version
```

> **NVIDIA GPU Users (Linux):** Ensure you have the NVIDIA driver and CUDA toolkit installed. On Pop!\_OS, run `sudo apt install system76-cuda-latest`. On Ubuntu, install the driver from `Additional Drivers` settings.

---

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
