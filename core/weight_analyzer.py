"""
Weight Analyzer
===============
Analyzes GGUF model tensors to understand their functional roles
and statistical properties. This information drives the Sorting
Algorithm's cell classification decisions.

Key responsibilities:
- Classify tensors by architectural role (attention, FFN, embedding, etc.)
- Compute statistical profiles (magnitude, variance, sparsity)
- Map tensors to layer zones (linguistic, semantic, reasoning)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from core.gguf_parser import GGUFFile, TensorInfo

logger = logging.getLogger(__name__)


@dataclass
class TensorProfile:
    """Statistical and functional profile of a single tensor."""
    tensor_info: TensorInfo
    # Functional classification
    layer_idx: int = -1              # -1 for non-layer tensors
    component: str = "unknown"       # attn_q, attn_k, ffn_gate, etc.
    functional_group: str = "CORE"   # CORE, ATTENTION, FFN
    layer_zone: str = "core"         # linguistic, semantic, reasoning, core
    # Statistical profile
    magnitude_mean: float = 0.0
    magnitude_std: float = 0.0
    sparsity: float = 0.0           # % of near-zero values
    variance: float = 0.0
    # Cell assignment (filled by SortingAlgorithm)
    cell_id: str = ""
    dna_tag: str = ""

    @property
    def is_core(self) -> bool:
        return self.functional_group == "CORE"

    @property
    def is_attention(self) -> bool:
        return self.functional_group == "ATTENTION"

    @property
    def is_ffn(self) -> bool:
        return self.functional_group == "FFN"


# Tensor name → component mapping patterns
TENSOR_COMPONENT_MAP = {
    # Core tensors
    "token_embd": ("token_embedding", "CORE", "core"),
    "output_norm": ("output_norm", "CORE", "core"),
    "output.weight": ("output_projection", "CORE", "core"),
    # Attention tensors
    "attn_q": ("attn_q", "ATTENTION", None),
    "attn_k": ("attn_k", "ATTENTION", None),
    "attn_v": ("attn_v", "ATTENTION", None),
    "attn_output": ("attn_output", "ATTENTION", None),
    "attn_norm": ("attn_norm", "ATTENTION", None),
    # FFN tensors
    "ffn_gate": ("ffn_gate", "FFN", None),
    "ffn_up": ("ffn_up", "FFN", None),
    "ffn_down": ("ffn_down", "FFN", None),
    "ffn_norm": ("ffn_norm", "FFN", None),
}


def _get_layer_zone(layer_idx: int, n_layers: int = 24) -> str:
    """Determine the functional zone of a layer."""
    third = n_layers // 3
    if layer_idx < third:
        return "linguistic"
    elif layer_idx < 2 * third:
        return "semantic"
    else:
        return "reasoning"


def _parse_tensor_name(name: str) -> tuple:
    """
    Parse a GGUF tensor name into (layer_idx, component).

    Examples:
        'blk.0.attn_q.weight' → (0, 'attn_q')
        'blk.15.ffn_gate.weight' → (15, 'ffn_gate')
        'token_embd.weight' → (-1, 'token_embd')
        'output_norm.weight' → (-1, 'output_norm')
    """
    # Block tensors: blk.{N}.{component}.weight
    if name.startswith("blk."):
        parts = name.split(".")
        layer_idx = int(parts[1])
        component = parts[2]
        return layer_idx, component

    # Non-block tensors
    base = name.replace(".weight", "").replace(".bias", "")
    return -1, base


class WeightAnalyzer:
    """
    Analyzes model weights to determine functional roles and statistical properties.

    This is the first step in the fragmentation pipeline:
    GGUF File → WeightAnalyzer → [TensorProfile, ...] → SortingAlgorithm
    """

    def __init__(self, gguf_file: GGUFFile, n_layers: int = 24):
        self.gguf_file = gguf_file
        self.n_layers = n_layers
        self.profiles: list[TensorProfile] = []

    def analyze(self, mm=None) -> list[TensorProfile]:
        """
        Analyze all tensors in the GGUF file.

        Args:
            mm: Optional memory-mapped file for statistical analysis.
                If None, only structural classification is performed.

        Returns:
            List of TensorProfile objects.
        """
        logger.info(f"Analyzing {len(self.gguf_file.tensors)} tensors...")
        self.profiles = []

        for tensor in self.gguf_file.tensors:
            profile = self._classify_tensor(tensor)

            # Compute statistics if memory map is available
            if mm is not None:
                self._compute_statistics(profile, tensor, mm)

            self.profiles.append(profile)

        self._log_summary()
        return self.profiles

    def _classify_tensor(self, tensor: TensorInfo) -> TensorProfile:
        """Classify a tensor based on its name and position."""
        layer_idx, component_key = _parse_tensor_name(tensor.name)

        # Look up component mapping
        component = component_key
        functional_group = "UNKNOWN"
        layer_zone = "core"

        for pattern, (comp, group, zone) in TENSOR_COMPONENT_MAP.items():
            if pattern in component_key:
                component = comp
                functional_group = group
                layer_zone = zone if zone else (
                    _get_layer_zone(layer_idx, self.n_layers)
                    if layer_idx >= 0 else "core"
                )
                break

        return TensorProfile(
            tensor_info=tensor,
            layer_idx=layer_idx,
            component=component,
            functional_group=functional_group,
            layer_zone=layer_zone,
        )

    def _compute_statistics(
        self,
        profile: TensorProfile,
        tensor: TensorInfo,
        mm,
    ):
        """Compute statistical properties of tensor weights."""
        try:
            # Only compute full stats for F32/F16 tensors
            if tensor.dtype_name in ("F32", "F16"):
                start = tensor.abs_offset
                end = start + tensor.size_bytes
                raw = mm[start:end]

                if tensor.dtype_name == "F32":
                    data = np.frombuffer(raw, dtype=np.float32)
                else:
                    data = np.frombuffer(raw, dtype=np.float16).astype(np.float32)

                profile.magnitude_mean = float(np.mean(np.abs(data)))
                profile.magnitude_std = float(np.std(np.abs(data)))
                profile.variance = float(np.var(data))

                # Sparsity: % of values with |x| < threshold
                threshold = profile.magnitude_mean * 0.01
                profile.sparsity = float(
                    np.sum(np.abs(data) < threshold) / len(data)
                )
            else:
                # For quantized tensors, use size-based heuristics
                profile.magnitude_mean = tensor.size_bytes / tensor.n_elements
                profile.variance = 0.0
                profile.sparsity = 0.0

        except Exception as e:
            logger.warning(
                f"Could not compute stats for {tensor.name}: {e}"
            )

    def _log_summary(self):
        """Log analysis summary."""
        groups = {}
        for p in self.profiles:
            key = p.functional_group
            groups[key] = groups.get(key, 0) + 1

        zones = {}
        for p in self.profiles:
            key = p.layer_zone
            zones[key] = zones.get(key, 0) + 1

        logger.info(f"Tensor groups: {groups}")
        logger.info(f"Layer zones: {zones}")

    def get_profiles_by_group(self, group: str) -> list[TensorProfile]:
        """Get all tensor profiles for a functional group."""
        return [p for p in self.profiles if p.functional_group == group]

    def get_profiles_by_zone(self, zone: str) -> list[TensorProfile]:
        """Get all tensor profiles for a layer zone."""
        return [p for p in self.profiles if p.layer_zone == zone]

    def get_profiles_by_layer(self, layer_idx: int) -> list[TensorProfile]:
        """Get all tensor profiles for a specific layer."""
        return [p for p in self.profiles if p.layer_idx == layer_idx]

    def get_core_profiles(self) -> list[TensorProfile]:
        """Get profiles that should always be loaded (core cells)."""
        return [p for p in self.profiles if p.is_core]
