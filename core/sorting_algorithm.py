"""
Sorting Algorithm (خوارزمية الفرز)
===================================
The core fragmentation engine that:
1. Analyzes weight tensors from the GGUF model
2. Classifies them into functional clusters
3. Writes independent cell files to disk
4. Generates a manifest for the Molecular Engine

Strategies:
-----------
- PER_LAYER: Each layer becomes one cell (simple, good baseline)
- PER_COMPONENT: Each component type across layers becomes a cell
- FUNCTIONAL: Group by function (attention vs FFN) × zone (linguistic/semantic/reasoning)
- HYBRID: Combine strategies for optimal granularity

The default FUNCTIONAL strategy creates cells like:
- Core: embedding, output_norm, output (always loaded)
- Attention cells per zone: linguistic_attn, semantic_attn, reasoning_attn
- FFN cells per zone: linguistic_ffn, semantic_ffn, reasoning_ffn
"""

import json
import logging
import mmap
import shutil
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

from core.gguf_parser import GGUFParser, GGUFFile, TensorInfo
from core.weight_analyzer import WeightAnalyzer, TensorProfile
from core.cell_taxonomy import (
    CellDefinition,
    CellManifest,
    generate_dna_tag,
)

logger = logging.getLogger(__name__)


class FragmentationStrategy(Enum):
    PER_LAYER = "per_layer"
    PER_COMPONENT = "per_component"
    FUNCTIONAL = "functional"
    HYBRID = "hybrid"


class SortingAlgorithm:
    """
    Main fragmentation engine.

    Pipeline:
        GGUF File → Parse → Analyze → Classify → Fragment → Write Cells
    """

    def __init__(
        self,
        gguf_path: str | Path,
        output_dir: str | Path,
        strategy: FragmentationStrategy = FragmentationStrategy.FUNCTIONAL,
        n_layers: int = 24,
    ):
        self.gguf_path = Path(gguf_path)
        self.output_dir = Path(output_dir)
        self.strategy = strategy
        self.n_layers = n_layers
        self.parser: Optional[GGUFParser] = None
        self.gguf_file: Optional[GGUFFile] = None
        self.profiles: list[TensorProfile] = []
        self.manifest: Optional[CellManifest] = None

    def execute(self) -> CellManifest:
        """
        Execute the full fragmentation pipeline.

        Returns:
            CellManifest with all cell definitions and metadata.
        """
        logger.info("=" * 60)
        logger.info("SORTING ALGORITHM — Atomic Model Fragmentation")
        logger.info(f"Strategy: {self.strategy.value}")
        logger.info(f"Input: {self.gguf_path}")
        logger.info(f"Output: {self.output_dir}")
        logger.info("=" * 60)

        # Step 1: Parse GGUF file
        self._parse_model()

        # Step 2: Analyze weights
        self._analyze_weights()

        # Step 3: Create cell definitions
        cells = self._create_cells()

        # Step 4: Write cell files to disk
        self._write_cells(cells)

        # Step 5: Generate and save manifest
        self._build_manifest(cells)

        logger.info("Fragmentation complete!")
        logger.info(f"Created {len(cells)} cells in {self.output_dir}")
        return self.manifest

    def _parse_model(self):
        """Step 1: Parse the GGUF model file."""
        logger.info("Step 1/5: Parsing GGUF model...")
        self.parser = GGUFParser(self.gguf_path)
        self.gguf_file = self.parser.parse()
        summary = self.gguf_file.summary()
        logger.info(
            f"  Model: {summary['model_name']} | "
            f"Tensors: {summary['n_tensors']} | "
            f"Size: {summary['total_size_mb']} MB"
        )

    def _analyze_weights(self):
        """Step 2: Analyze and classify all weight tensors."""
        logger.info("Step 2/5: Analyzing weight tensors...")
        analyzer = WeightAnalyzer(self.gguf_file, self.n_layers)

        # Memory map for statistical analysis
        mm = self.parser.memory_map(self.gguf_file)
        try:
            self.profiles = analyzer.analyze(mm)
        finally:
            mm.close()

        logger.info(f"  Analyzed {len(self.profiles)} tensor profiles")

    def _create_cells(self) -> list[CellDefinition]:
        """Step 3: Create cell definitions based on strategy."""
        logger.info(f"Step 3/5: Creating cells (strategy={self.strategy.value})...")

        if self.strategy == FragmentationStrategy.FUNCTIONAL:
            cells = self._strategy_functional()
        elif self.strategy == FragmentationStrategy.PER_LAYER:
            cells = self._strategy_per_layer()
        elif self.strategy == FragmentationStrategy.PER_COMPONENT:
            cells = self._strategy_per_component()
        elif self.strategy == FragmentationStrategy.HYBRID:
            cells = self._strategy_hybrid()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # Assign DNA tags
        for cell in cells:
            if not cell.dna_tag:
                cell.dna_tag = generate_dna_tag(
                    cell.functional_group,
                    cell.layer_zone,
                    cell.layer_indices[0] if cell.layer_indices else 0,
                    cell.components[0] if cell.components else "UNK",
                )

        logger.info(f"  Created {len(cells)} cell definitions")
        return cells

    def _strategy_functional(self) -> list[CellDefinition]:
        """
        FUNCTIONAL strategy: Group by (function × zone).

        Creates cells:
        - core_embedding, core_output_norm, core_output
        - linguistic_attn (layers 0-7 attention)
        - linguistic_ffn (layers 0-7 FFN)
        - semantic_attn (layers 8-15 attention)
        - semantic_ffn (layers 8-15 FFN)
        - reasoning_attn (layers 16-23 attention)
        - reasoning_ffn (layers 16-23 FFN)
        """
        cells = []

        # --- Core cells (always loaded) ---
        core_profiles = [p for p in self.profiles if p.is_core]
        if core_profiles:
            cell = CellDefinition(
                cell_id="core",
                dna_tag="C-X-000-ALL",
                functional_group="CORE",
                layer_zone="core",
                layer_indices=[-1],
                components=[p.component for p in core_profiles],
                tensor_names=[p.tensor_info.name for p in core_profiles],
                total_bytes=sum(p.tensor_info.size_bytes for p in core_profiles),
                n_tensors=len(core_profiles),
                always_loaded=True,
                priority=0,
            )
            cells.append(cell)

        # --- Zone cells (attention + FFN per zone) ---
        zones = {
            "linguistic": range(0, self.n_layers // 3),
            "semantic": range(self.n_layers // 3, 2 * self.n_layers // 3),
            "reasoning": range(2 * self.n_layers // 3, self.n_layers),
        }

        for zone_name, layer_range in zones.items():
            layers = list(layer_range)

            # Attention cell for this zone
            attn_profiles = [
                p for p in self.profiles
                if p.is_attention and p.layer_idx in layers
            ]
            if attn_profiles:
                cell = CellDefinition(
                    cell_id=f"{zone_name}_attn",
                    dna_tag=f"A-{zone_name[0].upper()}-{layers[0]:03d}-ALL",
                    functional_group="ATTENTION",
                    layer_zone=zone_name,
                    layer_indices=layers,
                    components=list(set(p.component for p in attn_profiles)),
                    tensor_names=[p.tensor_info.name for p in attn_profiles],
                    total_bytes=sum(
                        p.tensor_info.size_bytes for p in attn_profiles
                    ),
                    n_tensors=len(attn_profiles),
                    always_loaded=False,
                    priority=1 if zone_name == "linguistic" else 2,
                    depends_on=["core"],
                )
                cells.append(cell)

            # FFN cell for this zone
            ffn_profiles = [
                p for p in self.profiles
                if p.is_ffn and p.layer_idx in layers
            ]
            if ffn_profiles:
                cell = CellDefinition(
                    cell_id=f"{zone_name}_ffn",
                    dna_tag=f"F-{zone_name[0].upper()}-{layers[0]:03d}-ALL",
                    functional_group="FFN",
                    layer_zone=zone_name,
                    layer_indices=layers,
                    components=list(set(p.component for p in ffn_profiles)),
                    tensor_names=[p.tensor_info.name for p in ffn_profiles],
                    total_bytes=sum(
                        p.tensor_info.size_bytes for p in ffn_profiles
                    ),
                    n_tensors=len(ffn_profiles),
                    always_loaded=False,
                    priority=1 if zone_name == "linguistic" else 2,
                    depends_on=["core"],
                )
                cells.append(cell)

        return cells

    def _strategy_per_layer(self) -> list[CellDefinition]:
        """PER_LAYER strategy: One cell per transformer layer + core."""
        cells = []

        # Core cell
        core = [p for p in self.profiles if p.is_core]
        if core:
            cells.append(CellDefinition(
                cell_id="core",
                dna_tag="C-X-000-ALL",
                functional_group="CORE",
                layer_zone="core",
                layer_indices=[-1],
                components=[p.component for p in core],
                tensor_names=[p.tensor_info.name for p in core],
                total_bytes=sum(p.tensor_info.size_bytes for p in core),
                n_tensors=len(core),
                always_loaded=True,
                priority=0,
            ))

        # One cell per layer
        for layer_idx in range(self.n_layers):
            layer_profiles = [
                p for p in self.profiles if p.layer_idx == layer_idx
            ]
            if layer_profiles:
                zone = layer_profiles[0].layer_zone
                cells.append(CellDefinition(
                    cell_id=f"layer_{layer_idx:03d}",
                    dna_tag=f"B-{zone[0].upper()}-{layer_idx:03d}-ALL",
                    functional_group="BLOCK",
                    layer_zone=zone,
                    layer_indices=[layer_idx],
                    components=list(
                        set(p.component for p in layer_profiles)
                    ),
                    tensor_names=[
                        p.tensor_info.name for p in layer_profiles
                    ],
                    total_bytes=sum(
                        p.tensor_info.size_bytes for p in layer_profiles
                    ),
                    n_tensors=len(layer_profiles),
                    always_loaded=False,
                    priority=1,
                    depends_on=["core"],
                ))

        return cells

    def _strategy_per_component(self) -> list[CellDefinition]:
        """PER_COMPONENT strategy: Group same component across all layers."""
        cells = []

        # Core cell
        core = [p for p in self.profiles if p.is_core]
        if core:
            cells.append(CellDefinition(
                cell_id="core",
                dna_tag="C-X-000-ALL",
                functional_group="CORE",
                layer_zone="core",
                layer_indices=[-1],
                components=[p.component for p in core],
                tensor_names=[p.tensor_info.name for p in core],
                total_bytes=sum(p.tensor_info.size_bytes for p in core),
                n_tensors=len(core),
                always_loaded=True,
                priority=0,
            ))

        # Group by component type
        components = {}
        for p in self.profiles:
            if not p.is_core and p.component != "unknown":
                if p.component not in components:
                    components[p.component] = []
                components[p.component].append(p)

        for comp_name, profiles in components.items():
            group = profiles[0].functional_group
            cells.append(CellDefinition(
                cell_id=f"component_{comp_name}",
                dna_tag=generate_dna_tag(group, "core", 0, comp_name),
                functional_group=group,
                layer_zone="all",
                layer_indices=list(set(p.layer_idx for p in profiles)),
                components=[comp_name],
                tensor_names=[p.tensor_info.name for p in profiles],
                total_bytes=sum(
                    p.tensor_info.size_bytes for p in profiles
                ),
                n_tensors=len(profiles),
                always_loaded=False,
                priority=1,
                depends_on=["core"],
            ))

        return cells

    def _strategy_hybrid(self) -> list[CellDefinition]:
        """
        HYBRID strategy:
        - Core: always loaded
        - Attention: per-layer (fine-grained)
        - FFN: per-zone (coarse-grained, since FFN is larger)
        """
        cells = []

        # Core
        core = [p for p in self.profiles if p.is_core]
        if core:
            cells.append(CellDefinition(
                cell_id="core",
                dna_tag="C-X-000-ALL",
                functional_group="CORE",
                layer_zone="core",
                layer_indices=[-1],
                components=[p.component for p in core],
                tensor_names=[p.tensor_info.name for p in core],
                total_bytes=sum(p.tensor_info.size_bytes for p in core),
                n_tensors=len(core),
                always_loaded=True,
                priority=0,
            ))

        # Attention: per-layer
        for layer_idx in range(self.n_layers):
            attn = [
                p for p in self.profiles
                if p.is_attention and p.layer_idx == layer_idx
            ]
            if attn:
                zone = attn[0].layer_zone
                cells.append(CellDefinition(
                    cell_id=f"attn_{layer_idx:03d}",
                    dna_tag=f"A-{zone[0].upper()}-{layer_idx:03d}-ALL",
                    functional_group="ATTENTION",
                    layer_zone=zone,
                    layer_indices=[layer_idx],
                    components=list(set(p.component for p in attn)),
                    tensor_names=[p.tensor_info.name for p in attn],
                    total_bytes=sum(p.tensor_info.size_bytes for p in attn),
                    n_tensors=len(attn),
                    always_loaded=False,
                    priority=1,
                    depends_on=["core"],
                ))

        # FFN: per-zone (FUNCTIONAL style)
        zones = {
            "linguistic": range(0, self.n_layers // 3),
            "semantic": range(self.n_layers // 3, 2 * self.n_layers // 3),
            "reasoning": range(2 * self.n_layers // 3, self.n_layers),
        }
        for zone_name, layer_range in zones.items():
            layers = list(layer_range)
            ffn = [
                p for p in self.profiles
                if p.is_ffn and p.layer_idx in layers
            ]
            if ffn:
                cells.append(CellDefinition(
                    cell_id=f"{zone_name}_ffn",
                    dna_tag=f"F-{zone_name[0].upper()}-{layers[0]:03d}-ALL",
                    functional_group="FFN",
                    layer_zone=zone_name,
                    layer_indices=layers,
                    components=list(set(p.component for p in ffn)),
                    tensor_names=[p.tensor_info.name for p in ffn],
                    total_bytes=sum(p.tensor_info.size_bytes for p in ffn),
                    n_tensors=len(ffn),
                    always_loaded=False,
                    priority=2,
                    depends_on=["core"],
                ))

        return cells

    def _write_cells(self, cells: list[CellDefinition]):
        """Step 4: Write cell data files to disk."""
        logger.info("Step 4/5: Writing cell files to disk...")

        # Prepare output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Memory map the source GGUF for reading
        mm = self.parser.memory_map(self.gguf_file)

        try:
            for cell in cells:
                cell_path = self.output_dir / f"{cell.cell_id}.cell"
                self._write_single_cell(mm, cell, cell_path)
                logger.info(
                    f"  Written: {cell.cell_id}.cell "
                    f"({cell.size_mb:.2f} MB, {cell.n_tensors} tensors)"
                )
        finally:
            mm.close()

    def _write_single_cell(
        self,
        mm: mmap.mmap,
        cell: CellDefinition,
        cell_path: Path,
    ):
        """
        Write a single cell file.

        Cell file format:
        ┌──────────────────────────────┐
        │ Header (JSON, length-prefixed)│
        ├──────────────────────────────┤
        │ Tensor data (concatenated)    │
        └──────────────────────────────┘
        """
        # Build cell header with tensor offsets within the cell
        header_tensors = []
        current_offset = 0

        for tensor_name in cell.tensor_names:
            tensor = self.gguf_file.get_tensor(tensor_name)
            if tensor is None:
                logger.warning(f"Tensor not found: {tensor_name}")
                continue
            header_tensors.append({
                "name": tensor.name,
                "shape": list(tensor.shape),
                "dtype_id": tensor.dtype_id,
                "dtype_name": tensor.dtype_name,
                "offset_in_cell": current_offset,
                "size_bytes": tensor.size_bytes,
                "n_elements": tensor.n_elements,
            })
            current_offset += tensor.size_bytes

        header = {
            "cell_id": cell.cell_id,
            "dna_tag": cell.dna_tag,
            "functional_group": cell.functional_group,
            "layer_zone": cell.layer_zone,
            "n_tensors": len(header_tensors),
            "total_bytes": current_offset,
            "tensors": header_tensors,
        }

        header_json = json.dumps(header, ensure_ascii=False).encode("utf-8")
        header_length = len(header_json)

        with open(cell_path, "wb") as f:
            # Write header length (4 bytes, little-endian)
            f.write(header_length.to_bytes(4, byteorder="little"))
            # Write header JSON
            f.write(header_json)

            # Write tensor data
            for tensor_name in cell.tensor_names:
                tensor = self.gguf_file.get_tensor(tensor_name)
                if tensor is None:
                    continue
                start = tensor.abs_offset
                end = start + tensor.size_bytes
                # Copy from memory-mapped source
                f.write(mm[start:end])

    def _build_manifest(self, cells: list[CellDefinition]):
        """Step 5: Generate and save the cell manifest."""
        logger.info("Step 5/5: Building cell manifest...")

        self.manifest = CellManifest(
            model_name=self.gguf_file.metadata.get(
                "general.name", "unknown"
            ),
            model_architecture=self.gguf_file.metadata.get(
                "general.architecture", "unknown"
            ),
            total_cells=len(cells),
            total_bytes=sum(c.total_bytes for c in cells),
            cells=cells,
        )
        self.manifest.build_indices()

        manifest_path = self.output_dir / "manifest.json"
        self.manifest.save(manifest_path)

        summary = self.manifest.summary()
        logger.info(f"  Manifest: {json.dumps(summary, indent=2)}")

    def cleanup(self):
        """Clean up resources."""
        if self.parser:
            self.parser.close()
