"""
GGUF Parser
===========
Reads and parses GGUF (GGerganov's Unified Format) model files.
Provides zero-copy tensor access via memory-mapped I/O.

Architecture:
    GGUF File Layout:
    ┌─────────────┐
    │   Header     │  magic, version, tensor_count, metadata_count
    ├─────────────┤
    │  Metadata    │  key-value pairs (architecture, tokenizer, etc.)
    ├─────────────┤
    │ Tensor Info  │  name, shape, dtype, offset for each tensor
    ├─────────────┤
    │ Tensor Data  │  raw weight data (memory-mappable)
    └─────────────┘
"""

import struct
import mmap
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ============================================================
# GGUF Constants
# ============================================================
GGUF_MAGIC_LE = 0x46554747  # "GGUF" as little-endian u32 (bytes: 47 47 55 46)
GGUF_MAGIC_BE = 0x47475546  # "GGUF" as big-endian u32 (reversed)
GGUF_VERSION_3 = 3

# GGUF value types
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

# GGML tensor types and their byte sizes per element
GGML_TYPES = {
    0: ("F32", 4, np.float32),
    1: ("F16", 2, np.float16),
    2: ("Q4_0", 0.5 + 2/32, None),     # 4-bit quantized + scale
    3: ("Q4_1", 0.5 + 4/32, None),
    6: ("Q5_0", 0.625 + 2/32, None),
    7: ("Q5_1", 0.625 + 4/32, None),
    8: ("Q8_0", 1 + 2/32, None),
    9: ("Q8_1", 1 + 4/32, None),
    10: ("Q2_K", None, None),
    11: ("Q3_K", None, None),
    12: ("Q4_K", None, None),
    13: ("Q5_K", None, None),
    14: ("Q6_K", None, None),
    15: ("Q8_K", None, None),
    16: ("IQ2_XXS", None, None),
    17: ("IQ2_XS", None, None),
    18: ("IQ3_XXS", None, None),
    26: ("BF16", 2, None),
    30: ("F64", 8, np.float64),
}

# Block sizes for k-quant types
GGML_BLOCK_SIZES = {
    "Q2_K": 256,
    "Q3_K": 256,
    "Q4_K": 256,
    "Q5_K": 256,
    "Q6_K": 256,
    "Q8_K": 256,
    "Q4_0": 32,
    "Q4_1": 32,
    "Q5_0": 32,
    "Q5_1": 32,
    "Q8_0": 32,
    "Q8_1": 32,
}

# Type sizes for k-quant blocks (bytes per block)
GGML_TYPE_BLOCK_BYTES = {
    "Q2_K": 256 // 16 + 256 // 4 + 2 + 2,       # ~84 bytes for 256 elements
    "Q3_K": 256 // 8 * 3 + 256 // 16 + 2,        # ~110 bytes for 256 elements
    "Q4_K": 256 // 2 + 256 // 64 * 2 + 2 + 2,    # ~144 bytes for 256 elements
    "Q5_K": 256 // 8 * 5 + 256 // 64 * 2 + 2 + 2,  # ~176 bytes for 256 elements
    "Q6_K": 256 // 8 * 6 + 256 // 16 + 2,        # ~210 bytes for 256 elements
    "Q8_K": 256 + 4,                               # 260 bytes for 256 elements
    "Q4_0": 32 // 2 + 2,                           # 18 bytes for 32 elements
    "Q4_1": 32 // 2 + 2 + 2,                       # 20 bytes for 32 elements
    "Q5_0": 32 // 8 * 5 + 2,                       # 22 bytes for 32 elements
    "Q5_1": 32 // 8 * 5 + 2 + 2,                   # 24 bytes for 32 elements
    "Q8_0": 32 + 2,                                # 34 bytes for 32 elements
    "Q8_1": 32 + 4 + 4,                            # 40 bytes for 32 elements
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "F64": 8,
}


@dataclass
class TensorInfo:
    """Metadata about a single tensor in the GGUF file."""
    name: str
    n_dims: int
    shape: tuple
    dtype_id: int
    dtype_name: str
    offset: int           # Offset from start of data section
    abs_offset: int = 0   # Absolute offset in the file
    n_elements: int = 0
    size_bytes: int = 0

    def __post_init__(self):
        self.n_elements = 1
        for d in self.shape:
            self.n_elements *= d
        self._compute_size()

    def _compute_size(self):
        """Compute the size of this tensor's data in bytes."""
        type_name = self.dtype_name
        if type_name in ("F32", "F16", "BF16", "F64"):
            byte_per_elem = GGML_TYPE_BLOCK_BYTES.get(type_name, 4)
            self.size_bytes = self.n_elements * byte_per_elem
        elif type_name in GGML_BLOCK_SIZES:
            block_size = GGML_BLOCK_SIZES[type_name]
            n_blocks = self.n_elements // block_size
            block_bytes = GGML_TYPE_BLOCK_BYTES.get(type_name, block_size)
            self.size_bytes = n_blocks * block_bytes
        else:
            # Fallback estimate
            self.size_bytes = self.n_elements * 2  # Assume 2 bytes/elem


@dataclass
class GGUFFile:
    """Parsed representation of a GGUF file."""
    path: Path
    version: int = 0
    n_tensors: int = 0
    n_metadata: int = 0
    metadata: dict = field(default_factory=dict)
    tensors: list = field(default_factory=list)
    data_offset: int = 0  # Where tensor data begins
    alignment: int = 32

    @property
    def total_tensor_bytes(self) -> int:
        return sum(t.size_bytes for t in self.tensors)

    def get_tensor(self, name: str) -> Optional[TensorInfo]:
        """Find a tensor by name."""
        for t in self.tensors:
            if t.name == name:
                return t
        return None

    def get_tensors_by_layer(self, layer_idx: int) -> list:
        """Get all tensors belonging to a specific layer."""
        prefix = f"blk.{layer_idx}."
        return [t for t in self.tensors if t.name.startswith(prefix)]

    def get_tensors_by_pattern(self, pattern: str) -> list:
        """Get tensors matching a name pattern (substring)."""
        return [t for t in self.tensors if pattern in t.name]

    def summary(self) -> dict:
        """Return a summary of the parsed file."""
        return {
            "path": str(self.path),
            "version": self.version,
            "n_tensors": self.n_tensors,
            "n_metadata": self.n_metadata,
            "total_size_mb": round(self.total_tensor_bytes / (1024 * 1024), 2),
            "data_offset": self.data_offset,
            "architecture": self.metadata.get("general.architecture", "unknown"),
            "model_name": self.metadata.get("general.name", "unknown"),
        }


class GGUFParser:
    """
    Parser for GGUF model files.

    Reads the binary GGUF format and extracts:
    - File header (magic, version, counts)
    - Metadata key-value pairs
    - Tensor information (name, shape, type, data offset)

    Uses memory mapping for efficient zero-copy access to tensor data.
    """

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"GGUF file not found: {self.file_path}")
        self._fp = None
        self._mm = None
        self._pos = 0
        self._endian = "<"  # Default: little-endian

    def parse(self) -> GGUFFile:
        """Parse the GGUF file and return structured data."""
        logger.info(f"Parsing GGUF file: {self.file_path}")

        with open(self.file_path, "rb") as f:
            self._fp = f
            self._pos = 0

            # Parse header
            result = GGUFFile(path=self.file_path)
            self._parse_header(result)

            # Parse metadata
            self._parse_metadata(result)

            # Parse tensor info
            self._parse_tensor_info(result)

            # Compute data section offset (aligned)
            alignment = result.alignment
            data_start = self._pos
            if data_start % alignment != 0:
                data_start += alignment - (data_start % alignment)
            result.data_offset = data_start

            # Update tensor absolute offsets
            for tensor in result.tensors:
                tensor.abs_offset = result.data_offset + tensor.offset

        logger.info(
            f"Parsed {result.n_tensors} tensors, "
            f"{result.total_tensor_bytes / 1024 / 1024:.1f} MB total"
        )
        return result

    def memory_map(self, gguf_file: GGUFFile) -> mmap.mmap:
        """Create a memory-mapped view of the GGUF file for zero-copy tensor access."""
        f = open(gguf_file.path, "rb")
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        self._fp = f
        self._mm = mm
        return mm

    def read_tensor_data(
        self,
        mm: mmap.mmap,
        tensor: TensorInfo,
    ) -> np.ndarray:
        """
        Read raw tensor data from memory-mapped file.

        For F32/F16 tensors, returns a numpy array.
        For quantized tensors, returns raw bytes as uint8 array.
        """
        start = tensor.abs_offset
        end = start + tensor.size_bytes

        raw = mm[start:end]

        if tensor.dtype_name == "F32":
            return np.frombuffer(raw, dtype=np.float32).reshape(tensor.shape)
        elif tensor.dtype_name == "F16":
            return np.frombuffer(raw, dtype=np.float16).reshape(tensor.shape)
        else:
            # Quantized: return raw bytes for further processing
            return np.frombuffer(raw, dtype=np.uint8)

    def close(self):
        """Clean up memory mapping."""
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if self._fp is not None:
            self._fp.close()
            self._fp = None

    # --------------------------------------------------------
    # Private parsing methods
    # --------------------------------------------------------

    def _read(self, fmt: str) -> tuple:
        """Read and unpack binary data using detected endianness."""
        size = struct.calcsize(fmt)
        data = self._fp.read(size)
        if len(data) < size:
            raise EOFError(f"Unexpected end of file at position {self._pos}")
        self._pos += size
        return struct.unpack(f"{self._endian}{fmt}", data)

    def _read_u8(self) -> int:
        return self._read("B")[0]

    def _read_i8(self) -> int:
        return self._read("b")[0]

    def _read_u16(self) -> int:
        return self._read("H")[0]

    def _read_i16(self) -> int:
        return self._read("h")[0]

    def _read_u32(self) -> int:
        return self._read("I")[0]

    def _read_i32(self) -> int:
        return self._read("i")[0]

    def _read_u64(self) -> int:
        return self._read("Q")[0]

    def _read_i64(self) -> int:
        return self._read("q")[0]

    def _read_f32(self) -> float:
        return self._read("f")[0]

    def _read_f64(self) -> float:
        return self._read("d")[0]

    def _read_bool(self) -> bool:
        return self._read("?")[0]

    def _read_string(self) -> str:
        """Read a GGUF string (length-prefixed)."""
        length = self._read_u64()
        data = self._fp.read(length)
        if len(data) < length:
            raise EOFError("Unexpected end of file reading string")
        self._pos += length
        return data.decode("utf-8", errors="replace")

    def _read_value(self, type_id: int):
        """Read a GGUF value based on its type."""
        readers = {
            GGUF_TYPE_UINT8: self._read_u8,
            GGUF_TYPE_INT8: self._read_i8,
            GGUF_TYPE_UINT16: self._read_u16,
            GGUF_TYPE_INT16: self._read_i16,
            GGUF_TYPE_UINT32: self._read_u32,
            GGUF_TYPE_INT32: self._read_i32,
            GGUF_TYPE_UINT64: self._read_u64,
            GGUF_TYPE_INT64: self._read_i64,
            GGUF_TYPE_FLOAT32: self._read_f32,
            GGUF_TYPE_FLOAT64: self._read_f64,
            GGUF_TYPE_BOOL: self._read_bool,
            GGUF_TYPE_STRING: self._read_string,
        }

        if type_id == GGUF_TYPE_ARRAY:
            return self._read_array()

        reader = readers.get(type_id)
        if reader is None:
            raise ValueError(f"Unknown GGUF value type: {type_id}")
        return reader()

    def _read_array(self) -> list:
        """Read a GGUF array value."""
        elem_type = self._read_u32()
        count = self._read_u64()
        result = []
        for _ in range(count):
            result.append(self._read_value(elem_type))
        return result

    def _parse_header(self, result: GGUFFile):
        """Parse the GGUF file header with endianness detection."""
        # Read raw magic bytes (always use little-endian for first read)
        raw_magic = self._fp.read(4)
        if len(raw_magic) < 4:
            raise EOFError("File too small to contain GGUF header")
        self._pos += 4

        magic_le = struct.unpack("<I", raw_magic)[0]
        magic_be = struct.unpack(">I", raw_magic)[0]

        if magic_le == GGUF_MAGIC_LE:
            self._endian = "<"
            logger.debug("Detected little-endian GGUF")
        elif magic_le == GGUF_MAGIC_BE:
            self._endian = ">"
            logger.debug("Detected big-endian GGUF")
        else:
            # Show the actual bytes for debugging
            hex_bytes = " ".join(f"{b:02x}" for b in raw_magic)
            ascii_str = raw_magic.decode("ascii", errors="replace")
            raise ValueError(
                f"Invalid GGUF magic: 0x{magic_le:08x} (bytes: {hex_bytes}, "
                f"ascii: '{ascii_str}'). Expected GGUF format."
            )

        result.version = self._read_u32()
        if result.version < GGUF_VERSION_3:
            raise ValueError(
                f"Unsupported GGUF version: {result.version} "
                f"(minimum supported: {GGUF_VERSION_3})"
            )

        result.n_tensors = self._read_u64()
        result.n_metadata = self._read_u64()

        logger.debug(
            f"Header: version={result.version}, "
            f"tensors={result.n_tensors}, "
            f"metadata={result.n_metadata}"
        )

    def _parse_metadata(self, result: GGUFFile):
        """Parse all metadata key-value pairs."""
        for _ in range(result.n_metadata):
            key = self._read_string()
            type_id = self._read_u32()
            value = self._read_value(type_id)
            result.metadata[key] = value

            # Check for alignment override
            if key == "general.alignment":
                result.alignment = value

        logger.debug(f"Parsed {len(result.metadata)} metadata entries")

    def _parse_tensor_info(self, result: GGUFFile):
        """Parse tensor information entries."""
        for _ in range(result.n_tensors):
            name = self._read_string()
            n_dims = self._read_u32()
            shape = tuple(self._read_u64() for _ in range(n_dims))
            dtype_id = self._read_u32()
            offset = self._read_u64()

            dtype_name = "UNKNOWN"
            if dtype_id in GGML_TYPES:
                dtype_name = GGML_TYPES[dtype_id][0]

            tensor = TensorInfo(
                name=name,
                n_dims=n_dims,
                shape=shape,
                dtype_id=dtype_id,
                dtype_name=dtype_name,
                offset=offset,
            )
            result.tensors.append(tensor)

        logger.debug(f"Parsed {len(result.tensors)} tensor entries")


def find_gguf_model(model_name: str = "qwen2.5:0.5b") -> Optional[Path]:
    """
    Locate the GGUF file for a model installed via Ollama.

    Ollama stores models in:
      ~/.ollama/models/blobs/<sha256-hash>

    The mapping is stored in:
      ~/.ollama/models/manifests/registry.ollama.ai/library/<model>/<tag>
    """
    from config import OLLAMA_MODELS_DIR

    # Parse model name
    parts = model_name.split(":")
    name = parts[0]
    tag = parts[1] if len(parts) > 1 else "latest"

    # Read the manifest to find the blob hash
    manifest_path = (
        OLLAMA_MODELS_DIR / "manifests" / "registry.ollama.ai"
        / "library" / name / tag
    )

    if not manifest_path.exists():
        logger.warning(f"Model manifest not found: {manifest_path}")
        # Try to find any .gguf file in models directory
        models_dir = Path("d:/AI_NEW_GEN/models")
        gguf_files = list(models_dir.glob("*.gguf"))
        if gguf_files:
            return gguf_files[0]
        return None

    try:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        # Find the model layer (type: application/vnd.ollama.image.model)
        for layer in manifest.get("layers", []):
            if layer.get("mediaType") == "application/vnd.ollama.image.model":
                digest = layer["digest"]
                # Digest format: sha256:<hash>
                hash_value = digest.replace(":", "-")
                blob_path = OLLAMA_MODELS_DIR / "blobs" / hash_value
                if blob_path.exists():
                    return blob_path

        logger.warning("Model layer not found in manifest")
        return None
    except Exception as e:
        logger.error(f"Error reading manifest: {e}")
        return None
