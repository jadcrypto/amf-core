"""
AMF System Configuration
========================
Global configuration for the Atomic Model Fragmentation system.
"""

import os
from pathlib import Path

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).parent.resolve()
CELLS_DIR = PROJECT_ROOT / "cells"
MODELS_DIR = PROJECT_ROOT / "models"
MANIFEST_FILE = CELLS_DIR / "manifest.json"

# Ollama default model storage paths (Windows)
OLLAMA_MODELS_DIR = Path(os.environ.get(
    "OLLAMA_MODELS",
    Path.home() / ".ollama" / "models"
))

# ============================================================
# MODEL CONFIGURATION
# ============================================================
TARGET_MODEL = "qwen2.5:0.5b"
MODEL_ARCHITECTURE = "qwen2"

# Qwen2.5 0.5b architecture specs
MODEL_CONFIG = {
    "n_layers": 24,
    "n_embd": 896,
    "n_head": 14,
    "n_head_kv": 2,       # GQA: 2 KV heads
    "n_ff": 4864,          # FFN intermediate size
    "n_vocab": 151936,
    "context_length": 2048,  # Reduced for low VRAM
    "rope_theta": 1000000.0,
    "rms_norm_eps": 1e-6,
}

# ============================================================
# MEMORY CONSTRAINTS
# ============================================================
MAX_RAM_BUDGET_MB = 300          # Maximum RAM for loaded cells
MAX_VRAM_BUDGET_MB = 0           # No GPU usage (CPU only)
CORE_CELLS_BUDGET_MB = 80        # RAM reserved for always-loaded core cells
DYNAMIC_CELLS_BUDGET_MB = 200    # RAM for on-demand specialist cells
LRU_CACHE_SIZE = 10              # Number of cells to keep in LRU cache

# ============================================================
# CELL TAXONOMY
# ============================================================
# Functional categories for weight clusters
CELL_TYPES = {
    "CORE": {
        "description": "Essential cells always kept in memory",
        "subtypes": ["token_embedding", "output_norm", "output_projection"],
        "always_loaded": True,
    },
    "ATTENTION": {
        "description": "Self-attention mechanism cells",
        "subtypes": ["attn_q", "attn_k", "attn_v", "attn_output", "attn_norm"],
        "always_loaded": False,
    },
    "FFN": {
        "description": "Feed-forward network cells",
        "subtypes": ["ffn_gate", "ffn_up", "ffn_down", "ffn_norm"],
        "always_loaded": False,
    },
}

# Layer classification heuristic
# Early layers → linguistic/grammatical processing
# Middle layers → semantic understanding
# Late layers → reasoning/logic
LAYER_ZONES = {
    "linguistic": range(0, 8),      # Layers 0-7
    "semantic": range(8, 16),       # Layers 8-15
    "reasoning": range(16, 24),     # Layers 16-23
}

# ============================================================
# INTENT ANALYSIS
# ============================================================
INTENT_CATEGORIES = [
    "math_logic",
    "code_programming",
    "language_grammar",
    "creative_writing",
    "general_knowledge",
    "translation",
]

INTENT_CONFIDENCE_THRESHOLD = 0.85  # Minimum confidence to trigger pre-fetch
INTENT_REFINEMENT_WINDOW = 5        # Tokens to refine intent prediction

# ============================================================
# INFERENCE
# ============================================================
DEFAULT_MAX_TOKENS = 256
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
BATCH_SIZE = 1  # Single request at a time (resource constraint)

# ============================================================
# SYSTEM
# ============================================================
LOG_LEVEL = "INFO"
ENABLE_PROFILING = True  # Track memory/timing stats
