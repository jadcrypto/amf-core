"""
Cell Taxonomy
=============
Defines the classification system for model weight cells.
Each cell is a functional cluster of tensors that can be
independently loaded/unloaded from memory.

DNA Tagging:
    Each cell gets a unique DNA tag encoding:
    - Functional group (C=Core, A=Attention, F=FFN)
    - Layer zone (L=Linguistic, S=Semantic, R=Reasoning, X=Core)
    - Layer index
    - Component type

    Example: "A-L-003-Q" = Attention, Linguistic zone, Layer 3, Q-projection
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# DNA Tag encoding
GROUP_CODES = {
    "CORE": "C",
    "ATTENTION": "A",
    "FFN": "F",
}

ZONE_CODES = {
    "core": "X",
    "linguistic": "L",
    "semantic": "S",
    "reasoning": "R",
}

COMPONENT_CODES = {
    "token_embedding": "EMB",
    "output_norm": "ONR",
    "output_projection": "OUT",
    "attn_q": "Q",
    "attn_k": "K",
    "attn_v": "V",
    "attn_output": "O",
    "attn_norm": "AN",
    "ffn_gate": "G",
    "ffn_up": "U",
    "ffn_down": "D",
    "ffn_norm": "FN",
}


def generate_dna_tag(
    functional_group: str,
    layer_zone: str,
    layer_idx: int,
    component: str,
) -> str:
    """
    Generate a DNA tag for a tensor.

    Format: {GROUP}-{ZONE}-{LAYER:03d}-{COMPONENT}

    Examples:
        A-L-003-Q    → Attention Q-projection, linguistic layer 3
        F-R-020-G    → FFN gate, reasoning layer 20
        C-X-000-EMB  → Core embedding
    """
    g = GROUP_CODES.get(functional_group, "?")
    z = ZONE_CODES.get(layer_zone, "?")
    c = COMPONENT_CODES.get(component, component[:3].upper())
    layer = max(0, layer_idx)
    return f"{g}-{z}-{layer:03d}-{c}"


@dataclass
class CellDefinition:
    """Definition of a single functional cell."""
    cell_id: str                     # Unique identifier
    dna_tag: str                     # DNA tag
    functional_group: str            # CORE, ATTENTION, FFN
    layer_zone: str                  # linguistic, semantic, reasoning, core
    layer_indices: list = field(default_factory=list)
    components: list = field(default_factory=list)
    tensor_names: list = field(default_factory=list)
    # Size info
    total_bytes: int = 0
    n_tensors: int = 0
    # Cell properties
    always_loaded: bool = False      # Core cells
    priority: int = 0               # Loading priority (0=highest)
    # Dependencies
    depends_on: list = field(default_factory=list)  # Cell IDs this cell needs

    @property
    def size_mb(self) -> float:
        return self.total_bytes / (1024 * 1024)

    def to_dict(self) -> dict:
        return {
            "cell_id": self.cell_id,
            "dna_tag": self.dna_tag,
            "functional_group": self.functional_group,
            "layer_zone": self.layer_zone,
            "layer_indices": self.layer_indices,
            "components": self.components,
            "tensor_names": self.tensor_names,
            "total_bytes": self.total_bytes,
            "n_tensors": self.n_tensors,
            "always_loaded": self.always_loaded,
            "priority": self.priority,
            "depends_on": self.depends_on,
        }

    @staticmethod
    def from_dict(data: dict) -> "CellDefinition":
        return CellDefinition(**data)


@dataclass
class CellManifest:
    """
    Manifest of all cells produced by the Sorting Algorithm.
    Stored as manifest.json in the cells directory.
    """
    model_name: str = ""
    model_architecture: str = ""
    total_cells: int = 0
    total_bytes: int = 0
    cells: list = field(default_factory=list)  # List of CellDefinition

    # Layer zone → cell IDs mapping for quick lookup
    zone_index: dict = field(default_factory=dict)
    # Component → cell IDs mapping
    component_index: dict = field(default_factory=dict)

    def build_indices(self):
        """Build lookup indices for fast cell retrieval."""
        self.zone_index = {}
        self.component_index = {}

        for cell in self.cells:
            # Zone index
            zone = cell.layer_zone
            if zone not in self.zone_index:
                self.zone_index[zone] = []
            self.zone_index[zone].append(cell.cell_id)

            # Component index
            for comp in cell.components:
                if comp not in self.component_index:
                    self.component_index[comp] = []
                self.component_index[comp].append(cell.cell_id)

    def get_core_cells(self) -> list[CellDefinition]:
        """Get cells that should always be loaded."""
        return [c for c in self.cells if c.always_loaded]

    def get_cells_for_zone(self, zone: str) -> list[CellDefinition]:
        """Get all cells in a specific layer zone."""
        cell_ids = self.zone_index.get(zone, [])
        return [c for c in self.cells if c.cell_id in cell_ids]

    def get_cells_for_layers(self, layer_indices: list) -> list[CellDefinition]:
        """Get all cells that include any of the given layers."""
        result = []
        for cell in self.cells:
            if any(l in cell.layer_indices for l in layer_indices):
                result.append(cell)
        return result

    def save(self, path: Path):
        """Save manifest to JSON file."""
        data = {
            "model_name": self.model_name,
            "model_architecture": self.model_architecture,
            "total_cells": self.total_cells,
            "total_bytes": self.total_bytes,
            "cells": [c.to_dict() for c in self.cells],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Manifest saved: {path} ({self.total_cells} cells)")

    @staticmethod
    def load(path: Path) -> "CellManifest":
        """Load manifest from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        manifest = CellManifest(
            model_name=data["model_name"],
            model_architecture=data["model_architecture"],
            total_cells=data["total_cells"],
            total_bytes=data["total_bytes"],
            cells=[CellDefinition.from_dict(c) for c in data["cells"]],
        )
        manifest.build_indices()
        return manifest

    def summary(self) -> dict:
        """Return a summary of the manifest."""
        core_size = sum(c.total_bytes for c in self.cells if c.always_loaded)
        dynamic_size = sum(
            c.total_bytes for c in self.cells if not c.always_loaded
        )
        return {
            "model": self.model_name,
            "total_cells": self.total_cells,
            "total_size_mb": round(self.total_bytes / 1024 / 1024, 2),
            "core_cells": len(self.get_core_cells()),
            "core_size_mb": round(core_size / 1024 / 1024, 2),
            "dynamic_size_mb": round(dynamic_size / 1024 / 1024, 2),
            "zones": {
                zone: len(ids)
                for zone, ids in self.zone_index.items()
            },
        }
