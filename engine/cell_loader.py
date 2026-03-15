"""
Cell Loader
===========
Handles memory-mapped I/O for cell files.
Manages loading, unloading, and caching of weight cells.

Features:
- Zero-copy cell loading via mmap
- LRU cache for recently used cells
- Reference counting for safe unloading
- Memory pressure monitoring
"""

import json
import mmap
import logging
import struct
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CellTensorView:
    """A view into a loaded tensor within a cell."""
    name: str
    shape: tuple
    dtype_id: int
    dtype_name: str
    offset_in_cell: int
    size_bytes: int
    n_elements: int


@dataclass
class LoadedCell:
    """A cell that has been loaded into memory."""
    cell_id: str
    dna_tag: str
    functional_group: str
    layer_zone: str
    file_path: Path
    # Memory mapping
    file_handle: object = None
    mmap_handle: mmap.mmap = None
    data_offset: int = 0  # Where tensor data starts in the file
    # Tensor views
    tensors: list = field(default_factory=list)  # CellTensorView
    # State
    ref_count: int = 0
    total_bytes: int = 0

    def get_tensor_data(self, tensor_name: str) -> Optional[np.ndarray]:
        """
        Get raw tensor data as numpy array.

        For F32/F16: returns shaped array
        For quantized: returns flat uint8 array of raw bytes
        """
        for tv in self.tensors:
            if tv.name == tensor_name:
                start = self.data_offset + tv.offset_in_cell
                end = start + tv.size_bytes
                raw = self.mmap_handle[start:end]

                if tv.dtype_name == "F32":
                    return np.frombuffer(raw, dtype=np.float32).reshape(tv.shape)
                elif tv.dtype_name == "F16":
                    return np.frombuffer(raw, dtype=np.float16).reshape(tv.shape)
                else:
                    return np.frombuffer(raw, dtype=np.uint8)
        return None

    def get_all_tensors(self) -> dict:
        """Get all tensors as a name → numpy array dict."""
        result = {}
        for tv in self.tensors:
            data = self.get_tensor_data(tv.name)
            if data is not None:
                result[tv.name] = data
        return result

    @property
    def size_mb(self) -> float:
        return self.total_bytes / (1024 * 1024)

    def close(self):
        """Release memory mapping."""
        if self.mmap_handle is not None:
            try:
                self.mmap_handle.close()
            except Exception:
                pass
            self.mmap_handle = None
        if self.file_handle is not None:
            try:
                self.file_handle.close()
            except Exception:
                pass
            self.file_handle = None


class CellLoader:
    """
    Manages cell loading/unloading with LRU caching and memory tracking.

    Usage:
        loader = CellLoader(cells_dir, max_memory_mb=300)
        cell = loader.load("semantic_attn")
        data = cell.get_tensor_data("blk.10.attn_q.weight")
        loader.release("semantic_attn")
    """

    def __init__(
        self,
        cells_dir: str | Path,
        max_memory_mb: int = 300,
        lru_size: int = 10,
    ):
        self.cells_dir = Path(cells_dir)
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.lru_size = lru_size

        # Currently loaded cells
        self._loaded: OrderedDict[str, LoadedCell] = OrderedDict()
        # Memory tracking
        self._used_bytes: int = 0

    @property
    def used_memory_mb(self) -> float:
        return self._used_bytes / (1024 * 1024)

    @property
    def available_memory_mb(self) -> float:
        return (self.max_memory_bytes - self._used_bytes) / (1024 * 1024)

    @property
    def loaded_cell_ids(self) -> list[str]:
        return list(self._loaded.keys())

    def load(self, cell_id: str) -> Optional[LoadedCell]:
        """
        Load a cell into memory via mmap.

        If already loaded, increments reference count and returns it.
        If memory is insufficient, evicts LRU cells first.
        """
        # Already loaded?
        if cell_id in self._loaded:
            cell = self._loaded[cell_id]
            cell.ref_count += 1
            # Move to end (most recently used)
            self._loaded.move_to_end(cell_id)
            logger.debug(f"Cell {cell_id} already loaded (ref={cell.ref_count})")
            return cell

        # Find cell file
        cell_path = self.cells_dir / f"{cell_id}.cell"
        if not cell_path.exists():
            logger.error(f"Cell file not found: {cell_path}")
            return None

        # Read cell header to check size
        header, data_offset = self._read_cell_header(cell_path)
        if header is None:
            return None

        cell_bytes = header.get("total_bytes", 0)

        # Evict cells if needed to make room
        self._ensure_memory(cell_bytes)

        # Memory-map the cell file
        try:
            fh = open(cell_path, "rb")
            mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)

            # Build tensor views
            tensors = []
            for t in header.get("tensors", []):
                tensors.append(CellTensorView(
                    name=t["name"],
                    shape=tuple(t["shape"]),
                    dtype_id=t["dtype_id"],
                    dtype_name=t["dtype_name"],
                    offset_in_cell=t["offset_in_cell"],
                    size_bytes=t["size_bytes"],
                    n_elements=t["n_elements"],
                ))

            loaded = LoadedCell(
                cell_id=cell_id,
                dna_tag=header.get("dna_tag", ""),
                functional_group=header.get("functional_group", ""),
                layer_zone=header.get("layer_zone", ""),
                file_path=cell_path,
                file_handle=fh,
                mmap_handle=mm,
                data_offset=data_offset,
                tensors=tensors,
                ref_count=1,
                total_bytes=cell_bytes,
            )

            self._loaded[cell_id] = loaded
            self._used_bytes += cell_bytes

            logger.info(
                f"Loaded cell: {cell_id} "
                f"({loaded.size_mb:.2f} MB, {len(tensors)} tensors) "
                f"[Memory: {self.used_memory_mb:.1f}/{self.max_memory_bytes/1024/1024:.0f} MB]"
            )
            return loaded

        except Exception as e:
            logger.error(f"Failed to load cell {cell_id}: {e}")
            return None

    def release(self, cell_id: str):
        """Decrement reference count. Cell stays cached for LRU reuse."""
        if cell_id in self._loaded:
            self._loaded[cell_id].ref_count -= 1
            logger.debug(
                f"Released cell: {cell_id} "
                f"(ref={self._loaded[cell_id].ref_count})"
            )

    def unload(self, cell_id: str, force: bool = False):
        """
        Unload a cell from memory.

        Args:
            cell_id: Cell to unload
            force: If True, unload even if ref_count > 0
        """
        if cell_id not in self._loaded:
            return

        cell = self._loaded[cell_id]
        if cell.ref_count > 0 and not force:
            logger.warning(
                f"Cannot unload {cell_id}: ref_count={cell.ref_count}"
            )
            return

        cell.close()
        self._used_bytes -= cell.total_bytes
        del self._loaded[cell_id]
        logger.info(f"Unloaded cell: {cell_id}")

    def _ensure_memory(self, needed_bytes: int):
        """Evict LRU cells until we have enough memory."""
        while (
            self._used_bytes + needed_bytes > self.max_memory_bytes
            and self._loaded
        ):
            # Find oldest cell with ref_count == 0
            evict_id = None
            for cid, cell in self._loaded.items():
                if cell.ref_count <= 0 and not cell.dna_tag.startswith("C"):
                    evict_id = cid
                    break

            if evict_id is None:
                logger.warning(
                    f"Cannot free memory: all cells are in use or core. "
                    f"Need {needed_bytes/1024/1024:.1f} MB, "
                    f"used {self.used_memory_mb:.1f} MB"
                )
                break

            self.unload(evict_id, force=True)

    def _read_cell_header(self, cell_path: Path) -> tuple:
        """Read the JSON header from a cell file."""
        try:
            with open(cell_path, "rb") as f:
                # Read header length (4 bytes, little-endian)
                length_bytes = f.read(4)
                header_length = struct.unpack("<I", length_bytes)[0]
                # Read header JSON
                header_json = f.read(header_length)
                header = json.loads(header_json.decode("utf-8"))
                data_offset = 4 + header_length
                return header, data_offset
        except Exception as e:
            logger.error(f"Failed to read cell header from {cell_path}: {e}")
            return None, 0

    def get_stats(self) -> dict:
        """Get current loader statistics."""
        return {
            "loaded_cells": len(self._loaded),
            "used_memory_mb": round(self.used_memory_mb, 2),
            "available_memory_mb": round(self.available_memory_mb, 2),
            "max_memory_mb": round(self.max_memory_bytes / 1024 / 1024, 2),
            "cells": {
                cid: {
                    "size_mb": round(cell.size_mb, 2),
                    "ref_count": cell.ref_count,
                    "n_tensors": len(cell.tensors),
                    "dna_tag": cell.dna_tag,
                }
                for cid, cell in self._loaded.items()
            },
        }

    def close_all(self):
        """Unload all cells and clean up."""
        for cell_id in list(self._loaded.keys()):
            self.unload(cell_id, force=True)
        logger.info("All cells unloaded")
