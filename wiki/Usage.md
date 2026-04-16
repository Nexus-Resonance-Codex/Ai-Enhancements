# Usage Guide

Welcome to the tactical ledger for the 256D Lattice Visualizer. This guide provides the standardized protocols for interacting with the holographic explorer, projecting custom vectors, and navigating the φ-tensor manifold with institutional precision.

## Interacting with the Holographic Explorer

The visualizer is designed as a dynamic, low-latency window into high-dimensional space. Effective exploration requires a synchronization of keyboard and mouse inputs to navigate the recursive lattice and isolate 7-stable resonance anchors.

### 🎮 1-2-4-5-7 Interaction Protocol

Follow these strictly phased steps to master the visual manifold:

1.  **Launch the Gateway**:
    Initialize the visualizer kernel and open the Gradio landing page.
    ```bash
    uv run python -m nrc.visualizer.app
    ```

2.  **Focus the Locus**:
    Use the **F** key to automatically align the camera with the primary TTT_7 resonance locus.

4.  **Project Custom Tensors**:
    Utilize the Python API (LatticeClient) to inject your research vectors into the active viewport.
    ```python
    from nrc.visualizer import LatticeClient
    client = LatticeClient()
    client.project(tensor_data, label="Research_Vector_Alpha")
    ```

5.  **Shift Dimensions**:
    Dynamically transition between 256D and 729D structural phasing to observe emergent complexity.

7.  **Certify the Frame**:
    Capture a high-resolution holographic snapshot and export the structural metadata to the `exports/` manifold.

### ⌨️ Advanced Control Map

| Control | Action | Phasing Impact |
| :--- | :--- | :--- |
| **Space** | Toggle Lattice Rotation | 🟢 Neutral |
| **R** | Reset Camera to Origin | 🔵 Stable |
| **F** | Focus on 7-Stable Locus | 🟡 Active |
| **S** | Save High-Res Snapshot | ⚪ Logged |
| **Mouse Wheel** | Zoom In/Out | 🔘 Scaling |

---

### 📸 Tactical View: The 7-Stable Focus
![Visualizer Focus Mode](https://raw.githubusercontent.com/Nexus-Resonance-Codex/NRC/main/visualizations/visualizer_focus_mode.png)
*Figure 4: The holographic explorer in Focus Mode, isolating the TTT_7 resonance anchor within a 729D projection manifold.*


## Advanced Projection and Manifold Mapping

Beyond basic navigation, the visualizer allows for the direct injection of research vectors from the `NRC` core and `Protein-Folding` manifolds. This provides a direct physical verification of computed states within the 729D lattice, enabling researchers to observe emergent convergence in real-time.

### 🐍 1-2-4-5-7 Remote API Sequence

Use the `LatticeClient` to synchronize your external research environment with the holographic visualizer:

1.  **Instantiate the Lattice Client**:
    Establish a remote handshake with the local or distributed visualizer vertex.
    ```python
    from nrc.visualizer import LatticeClient
    client = LatticeClient(host="localhost", port=7860)
    ```

2.  **Prepare the Phasing Tensor**:
    Normalize your high-dimensional research vector for absolute golden-basis alignment.
    ```python
    import numpy as np
    # Generate a normalized 256D resonance vector
    vector = np.random.randn(256) * 1.618033
    ```

4.  **Inject into the Manifold**:
    Project the vector into the holographic space with a TTT-compliant lighthouse label.
    ```python
    client.project(vector, label="E8_Resonance_Alpha_7")
    ```

5.  **Apply QRT Damping Filter**:
    Visually dampen the residue turbulence to isolate the core structural stability.
    ```python
    client.set_damping(mode="QRT", intensity=0.3819)
    ```

7.  **Synchronize Perspective**:
    Force the institutional 7-stable camera focus to observe the newly injected point.
    ```python
    client.focus_on("E8_Resonance_Alpha_7")
    ```

### 💾 Data Persistence and Exports

The visualizer supports the extraction of high-integrity structural data for further analysis in the `Phi-Infinity` or `Protein-Folding` modules.

*   **OBJ Export**: Full holographic mesh for external 3D analysis and 3D printing of resonance states.
*   **JSON Residue**: Raw state vector coordinates and TTT-phasing metadata for archival analysis.
*   **PNG Spectrum**: High-dynamic-range color spectral maps for publication-grade visual reporting.
*   **YAML Metadata**: Institutional-grade log of all phasing transformations and QRT damping levels.
*   **PPD Mapping**: Binary projection data compatible with the Protein Resonance Accelerator.

---

### 📸 Prototyping: Custom Lattice Injection
![Custom Projection](https://raw.githubusercontent.com/Nexus-Resonance-Codex/NRC/main/visualizations/custom_projection_example.png)
*Figure 5: High-resolution projection of a custom dihedral protein manifold showing the emergent TTT_7 stability shell around an unknown structural seed.*

### ⏭️ Next Steps

Phasing complete. For full technical specifications, proceed to the **[API Reference](API-Reference.md)** or explore the **[Mathematical Foundations](Home.md#core-philosophy-and-mathematical-foundations)** for the 729D projection derivation.

---
← [Back to Core Home](../../NRC/wiki/Home.md) | [Back to Visualizer Home](Home.md) | [Table of Contents](Home.md#project-overview) | [Back to Top](#usage-guide)
