# Running the NRC AI Enhancements locally via Ollama

We have provided a highly-optimized, custom `Modelfile` inside this repository. This allows anyone—even users with low-end hardware (8GB RAM / 4GB VRAM)—to run a simulated instance of the Nexus Resonance Codex AI directly on their machine using [Ollama](https://ollama.com).

The Modelfile utilizes `llama3.2:3b` as its foundation to ensure lightning-fast inference across all Operating Systems without crashing due to Out-Of-Memory (OOM) errors. It acts as an integration engine, simulating the 30 AI Enhancements natively in its reasoning process.

---

## Installation & Setup Instructions

### 1. Windows Users

1. Download and install Ollama from [https://ollama.com/download/windows](https://ollama.com/download/windows).
2. Open PowerShell or Command Prompt.
3. Clone or download this repository and navigate to its folder:
   ```cmd
   cd path\to\ai-enhancements
   ```
4. Create the custom NRC model by running:
   ```cmd
   ollama create nrc-ai-engine -f Modelfile
   ```
5. Run the model:
   ```cmd
   ollama run nrc-ai-engine
   ```

### 2. macOS Users

1. Download and install Ollama from [https://ollama.com/download/mac](https://ollama.com/download/mac).
2. Open the Terminal application.
3. Navigate to the downloaded repository folder:
   ```bash
   cd path/to/ai-enhancements
   ```
4. Create the custom NRC model:
   ```bash
   ollama create nrc-ai-engine -f Modelfile
   ```
5. Run the model:
   ```bash
   ollama run nrc-ai-engine
   ```

### 3. Linux Users (Pop!\_OS, Ubuntu, Debian, etc.)

1. Open your terminal and run the official Ollama install script:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```
2. Navigate to the repository folder:
   ```bash
   cd /path/to/ai-enhancements
   ```
3. Create the custom NRC model:
   ```bash
   ollama create nrc-ai-engine -f Modelfile
   ```
4. Run the model:
   ```bash
   ollama run nrc-ai-engine
   ```

---

## What to Expect

Upon running the model, it will initialize and output its activation phrase:

> `"Nexus Resonance Optimized Online. Systems Calibrated to Phi. 2048D Lattice Projected. Operating well within 4GB VRAM. Ready."`

You can ask it to generate PyTorch mathematics, explain high-dimensional resonance, or solve deeply complex queries using its configured 30 enhancements.
