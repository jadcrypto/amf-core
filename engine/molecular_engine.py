"""
Molecular Engine (المحرك الجزيئي)
==================================
The central orchestrator of the AMF system.

Responsibilities:
- Manages cell lifecycle (loading, caching, unloading)
- Routes requests through Intent Analysis → Cell Selection → Inference
- Maintains core cells always in memory
- Implements Micro-MoE routing
- Handles pre-fetching based on intent prediction
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from core.cell_taxonomy import CellManifest, CellDefinition
from engine.cell_loader import CellLoader, LoadedCell
from engine.intent_analyzer import IntentAnalyzer, IntentResult

logger = logging.getLogger(__name__)


@dataclass
class EngineStats:
    """Runtime statistics for the engine."""
    total_requests: int = 0
    total_cells_loaded: int = 0
    total_cells_evicted: int = 0
    avg_cells_per_request: float = 0.0
    avg_response_time_ms: float = 0.0
    peak_memory_mb: float = 0.0
    # Per-request tracking
    _cell_counts: list = field(default_factory=list)
    _response_times: list = field(default_factory=list)

    def record_request(self, n_cells: int, time_ms: float):
        self.total_requests += 1
        self._cell_counts.append(n_cells)
        self._response_times.append(time_ms)
        self.avg_cells_per_request = sum(self._cell_counts) / len(self._cell_counts)
        self.avg_response_time_ms = sum(self._response_times) / len(self._response_times)

    def to_dict(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "avg_cells_per_request": round(self.avg_cells_per_request, 1),
            "avg_response_time_ms": round(self.avg_response_time_ms, 1),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
        }


class MolecularEngine:
    """
    The heart of the AMF system.

    Pipeline:
        Prompt → IntentAnalyzer → Cell Selection → CellLoader → Inference

    Usage:
        engine = MolecularEngine(cells_dir="cells/", manifest_path="cells/manifest.json")
        engine.initialize()
        result = engine.process("What is 2+2?")
        engine.shutdown()
    """

    def __init__(
        self,
        cells_dir: str | Path,
        manifest_path: str | Path,
        max_memory_mb: int = 300,
        lru_size: int = 10,
    ):
        self.cells_dir = Path(cells_dir)
        self.manifest_path = Path(manifest_path)
        self.max_memory_mb = max_memory_mb

        # Components
        self.manifest: Optional[CellManifest] = None
        self.cell_loader: Optional[CellLoader] = None
        self.intent_analyzer: Optional[IntentAnalyzer] = None

        # State
        self._initialized = False
        self.stats = EngineStats()

    def initialize(self):
        """
        Initialize the engine:
        1. Load cell manifest
        2. Create cell loader
        3. Load core cells (always in memory)
        4. Initialize intent analyzer
        """
        logger.info("=" * 50)
        logger.info("Molecular Engine — Initializing")
        logger.info("=" * 50)

        # Load manifest
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Cell manifest not found: {self.manifest_path}\n"
                "Run the Sorting Algorithm first to fragment the model."
            )

        self.manifest = CellManifest.load(self.manifest_path)
        logger.info(f"Loaded manifest: {self.manifest.total_cells} cells")

        # Initialize cell loader
        self.cell_loader = CellLoader(
            cells_dir=self.cells_dir,
            max_memory_mb=self.max_memory_mb,
        )

        # Load core cells
        core_cells = self.manifest.get_core_cells()
        for cell_def in core_cells:
            loaded = self.cell_loader.load(cell_def.cell_id)
            if loaded is None:
                logger.error(f"Failed to load core cell: {cell_def.cell_id}")
            else:
                logger.info(
                    f"Core cell loaded: {cell_def.cell_id} "
                    f"({loaded.size_mb:.2f} MB)"
                )

        # Initialize intent analyzer
        self.intent_analyzer = IntentAnalyzer()

        self._initialized = True
        logger.info("Molecular Engine — Ready")
        self._log_memory_status()

    def process(self, prompt: str) -> dict:
        """
        Process a user prompt through the full pipeline.

        Args:
            prompt: User's input text.

        Returns:
            Dict with:
                - intent: IntentResult
                - loaded_cells: list of cell IDs
                - tensors: dict of tensor_name → numpy array
                - stats: timing and memory info
        """
        if not self._initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        start_time = time.perf_counter()

        # Step 1: Analyze intent
        intent = self.intent_analyzer.analyze(prompt)
        logger.info(
            f"Intent: {intent.primary_intent} "
            f"(conf={intent.confidence:.2f})"
        )

        # Step 2: Determine required cells
        required_cell_ids = self.intent_analyzer.get_required_cell_ids(intent)
        logger.info(f"Required cells: {required_cell_ids}")

        # Step 3: Load required cells
        loaded_cells = self._load_required_cells(required_cell_ids)

        # Step 4: Collect all tensors from loaded cells
        all_tensors = {}
        for cell_id in loaded_cells:
            cell = self.cell_loader._loaded.get(cell_id)
            if cell:
                tensors = cell.get_all_tensors()
                all_tensors.update(tensors)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Update stats
        self.stats.record_request(len(loaded_cells), elapsed_ms)
        mem_mb = self.cell_loader.used_memory_mb
        if mem_mb > self.stats.peak_memory_mb:
            self.stats.peak_memory_mb = mem_mb

        result = {
            "intent": intent,
            "loaded_cells": loaded_cells,
            "n_tensors": len(all_tensors),
            "total_tensor_names": list(all_tensors.keys()),
            "elapsed_ms": round(elapsed_ms, 2),
            "memory_used_mb": round(mem_mb, 2),
        }

        logger.info(
            f"Processed in {elapsed_ms:.1f}ms | "
            f"Cells: {len(loaded_cells)} | "
            f"Tensors: {len(all_tensors)} | "
            f"Memory: {mem_mb:.1f} MB"
        )

        # Release non-core cells (they stay cached but ref_count decreases)
        for cell_id in loaded_cells:
            if cell_id != "core":
                self.cell_loader.release(cell_id)

        return result

    def get_loaded_tensors(self) -> dict:
        """Get all tensors currently in memory."""
        all_tensors = {}
        for cell_id, cell in self.cell_loader._loaded.items():
            tensors = cell.get_all_tensors()
            all_tensors.update(tensors)
        return all_tensors

    def _load_required_cells(self, cell_ids: list[str]) -> list[str]:
        """Load all required cells, handling dependencies."""
        loaded = []
        for cell_id in cell_ids:
            # Check if cell exists in manifest
            cell_def = None
            for c in self.manifest.cells:
                if c.cell_id == cell_id:
                    cell_def = c
                    break

            if cell_def is None:
                logger.warning(f"Cell not found in manifest: {cell_id}")
                continue

            # Load dependencies first
            for dep_id in cell_def.depends_on:
                if dep_id not in loaded:
                    result = self.cell_loader.load(dep_id)
                    if result:
                        loaded.append(dep_id)

            # Load the cell
            result = self.cell_loader.load(cell_id)
            if result:
                loaded.append(cell_id)

        return list(set(loaded))

    def get_stats(self) -> dict:
        """Get comprehensive engine statistics."""
        return {
            "engine": self.stats.to_dict(),
            "loader": self.cell_loader.get_stats() if self.cell_loader else {},
            "manifest": self.manifest.summary() if self.manifest else {},
        }

    def _log_memory_status(self):
        """Log current memory usage."""
        if self.cell_loader:
            stats = self.cell_loader.get_stats()
            logger.info(
                f"Memory: {stats['used_memory_mb']:.1f} / "
                f"{stats['max_memory_mb']:.0f} MB | "
                f"Loaded: {stats['loaded_cells']} cells"
            )

    def shutdown(self):
        """Clean up all resources."""
        logger.info("Molecular Engine — Shutting down")
        if self.cell_loader:
            self.cell_loader.close_all()
        self._initialized = False
        logger.info("Molecular Engine — Stopped")
