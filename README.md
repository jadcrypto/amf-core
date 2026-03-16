# ⚛️ AMF — Atomic Model Fragmentation

<div align="center">

**First Arabic-originated Molecular Inference Engine for LLMs**
**أول مكتبة عربية توفر محرك استدلال جزيئي للنماذج اللغوية الكبيرة**

[![PyPI version](https://badge.fury.io/py/amf-core.svg)](https://badge.fury.io/py/amf-core)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![RAM](https://img.shields.io/badge/RAM--%3C%20500%20MB-green)](https://github.com/jadcrypto/amf-core)

```
pip install amf-core
```

</div>

---

## What is AMF?

AMF (Atomic Model Fragmentation) is a Python library that enables running large language models (LLMs) on **severely resource-constrained hardware** — including machines with less than 1 GB of VRAM and as little as 300 MB of RAM.

Instead of loading an entire model into memory, AMF:

1. **Fragments** the model into independent semantic *cells*
2. **Analyzes** each user prompt to determine which cells are needed
3. **Loads** only the required cells on demand via memory-mapping
4. **Infers** using only the active subset of weights

> **Validated on Kaggle:** Qwen2.5-7B (9 GB model) running with < 500 MB RAM ✅

---

## Installation

```bash
pip install amf-core
```

**Optional extras:**
```bash
pip install amf-core[safetensors]   # Safetensors / HuggingFace support
pip install amf-core[dev]           # Development tools
```

**Requirements:** Python ≥ 3.10, numpy, gguf, rich, scikit-learn

---

## Quick Start

### 1 — Direct Molecular Inference (fastest)

```python
import amf

eng = amf.engine("path/to/model.gguf")
eng.load()

print(eng.predict("Hello"))           # → "resilient"
print(eng.generate("Hello", n=5))    # → "Hello resilient strong bold clear"

eng.close()
```

### 2 — Context manager (auto-close)

```python
import amf

with amf.engine("model.gguf") as eng:
    print(eng.predict("The future of AI is"))
```

### 3 — Full Fragmentation Pipeline

```python
import amf

model = amf.load_universal("path/to/model.gguf")
cells = amf.fragment(model, strategy="functional", output_dir="./cells")
print(f"Generated {cells.total_cells} cells ({cells.total_bytes / 1e6:.0f} MB)")

for cell in cells.cells:
    print(f"  {cell.cell_id:30s}  DNA: {cell.dna_tag}  ({cell.size_mb:.1f} MB)")
```

### 4 — CLI

```bash
amf fragment --model qwen2.5:0.5b   # Fragment a model
amf chat                             # Interactive chat
amf info                             # System info
```

---

## Architecture

AMF operates in 6 distinct phases:

```
User Prompt
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│  1. Universal Parsing    │  ModelLoader reads GGUF /     │
│                          │  Safetensors → UniversalModel │
├──────────────────────────┼───────────────────────────────┤
│  2. Weight Analysis      │  WeightAnalyzer classifies    │
│                          │  tensors by zone & function   │
├──────────────────────────┼───────────────────────────────┤
│  3. DNA Tagging          │  SortingAlgorithm assigns     │
│                          │  tags: A-L-003-Q              │
│                          │  (Attention-Linguistic-L3-Q)  │
├──────────────────────────┼───────────────────────────────┤
│  4. Intent Analysis      │  IntentAnalyzer determines    │
│                          │  which layer zones are needed │
├──────────────────────────┼───────────────────────────────┤
│  5. Molecular Engine     │  Loads only required cells    │
│                          │  via selective mmap           │
├──────────────────────────┼───────────────────────────────┤
│  6. AMFEngine Inference  │  Forward pass on active       │
│                          │  weight subset (NumPy)        │
└──────────────────────────┴───────────────────────────────┘
```

### Layer Zones

| Zone       | Layers | Specialisation          |
|------------|--------|-------------------------|
| Linguistic | 0 – 7  | Grammar, syntax, tokens |
| Semantic   | 8 – 15 | Meaning, context        |
| Reasoning  | 16–23  | Logic, math, code       |

### DNA Tag Format

```
A  -  L  -  003  -  Q
│     │      │      └─ Component  (Q/K/V/O/Gate/Up/Down/Norm)
│     │      └──────── Layer index (zero-padded)
│     └─────────────── Zone       (L=Linguistic / S=Semantic / R=Reasoning)
└───────────────────── Type       (A=Attention / F=FFN / C=Core)
```

---

## AMFEngine API

```python
from engine.amf_engine import AMFEngine

eng = AMFEngine("model.gguf", inference_layer=20)
eng.load()

eng.predict("Hello")          # → str
eng.generate("Hello", n=5)   # → str
eng.set_layer(16)             # switch inference layer
eng.list_layers()             # → [0, 1, ..., 23]
eng.info()                    # → dict
eng.close()
```

### Supported quantisation formats

| Format | Status |
|--------|--------|
| F32    | ✅ Full |
| F16    | ✅ Full |
| Q8_0   | ✅ Full |
| Q4_K_M | ✅ Approximate |
| Q4_0   | ✅ Approximate |

---

## Benchmark

| Model            | Full RAM | AMF RAM | Reduction |
|------------------|----------|---------|-----------|
| Qwen2.5-0.5B     | ~400 MB  | ~80 MB  | 5×        |
| Qwen2.5-7B (Q4)  | ~4.5 GB  | ~500 MB | 9×        |
| Qwen2.5-32B (Q4) | ~20 GB   | ~500 MB | 40×       |

> CPU-only. No GPU required.

---

## Supported Models

| Model Family         | Format      | Status        |
|----------------------|-------------|---------------|
| Qwen 2.5 (all sizes) | GGUF        | ✅ Tested     |
| Qwen 3 / 3.5         | GGUF        | ✅ Compatible |
| LLaMA 3              | GGUF        | ✅ Compatible |
| Mistral              | GGUF        | ✅ Compatible |
| HuggingFace models   | Safetensors | 🔄 Beta       |

---

## Contributing

```bash
git clone https://github.com/jadcrypto/amf-core.git
cd amf-core
pip install -e .[dev]
pytest tests/
```

---

## License

MIT — see [LICENSE](LICENSE).

---

## Citation

```bibtex
@software{amf_core,
  title  = {AMF: Atomic Model Fragmentation},
  author = {Jad},
  year   = {2026},
  url    = {https://github.com/jadcrypto/amf-core}
}
```

---

<div align="center">
Built with ❤️ — First molecular inference engine from the Arab world
</div>
