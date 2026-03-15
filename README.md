# Atomic Model Fragmentation (AMF)

**المحرك الجزيئي — Molecular Engine**

A revolutionary approach to running Large Language Models on resource-constrained hardware by breaking them down into functional, on-demand cells.

## Installation

```bash
# Clone the repository
cd AI_NEW_GEN

# Install as a library (development mode)
pip install -e .
```

## Quick Start (CLI)

1. **Fragment a model:**

   ```bash
   amf fragment --model qwen2.5:0.5b
   ```

2. **Start Interactive Engine:**

   ```bash
   amf chat
   ```

## Python Library Usage (`amf-core`)

The AMF system provides a universal Python library abstraction allowing you to easily embed this technology in any Python application.

```python
import amf

# 1. Load Universal Model (GGUF, Safetensors auto-detection)
model = amf.load_universal("/path/to/model.gguf")

# 2. Apply Genetic Fragmentation Strategy
cells = amf.fragment(
    model, 
    strategy="functional", 
    output_dir="./cells"
)

# 3. Explore Cellular DNA
for cell in cells.cells:
    print(f"Cell ID: {cell.cell_id}")
    print(f"DNA Tag: {cell.dna_tag}")
    print(f"Size: {cell.size_mb:.1f} MB")
```

## System Architecture

AMF operates in 6 distinct phases:

1. **Universal Parsing**: `ModelLoader` reads formats like GGUF and translates them into a universal tensor abstraction.
2. **Weight Analysis**: `WeightAnalyzer` classifies weights by layer zone, functional role, and computes statistics.
3. **DNA Tagging & Fragmentation**: `SortingAlgorithm` chunks weights into semantic groups (e.g., `A-L-003-Q` for Attention-Linguistic-Layer3-QBlock).
4. **Intent Analysis**: `IntentAnalyzer` parses user prompts to determine required reasoning/semantic layer zones.
5. **Molecular Orchestration**: `MolecularEngine` triggers selective `mmap` loading of only the necessary cells into memory.
6. **Dynamic Inference**: `InferenceEngine` executes the forward pass on the loaded functional subset using NumPy.
