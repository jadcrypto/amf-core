"""
Universal Model Loader
======================
Provides a unified interface to load different model formats
(GGUF, Safetensors, PyTorch) into a standard memory representation
for the Atomic Model Fragmentation (AMF) system.
"""

import logging
from pathlib import Path
from typing import Optional, Protocol, Any

from core.gguf_parser import GGUFParser, GGUFFile

logger = logging.getLogger(__name__)


class UniversalModel(Protocol):
    """Protocol defining the standard interface for loaded models."""
    
    @property
    def architecture(self) -> str: ...
    
    @property
    def total_size_bytes(self) -> int: ...
    
    @property
    def tensors(self) -> list: ...
    
    def summary(self) -> dict: ...


# Wrapper to make GGUFFile implement the UniversalModel protocol
class GGUFUniversalAdapter:
    def __init__(self, gguf_file: GGUFFile, parser: GGUFParser):
        self.gguf = gguf_file
        self.parser = parser
        
    @property
    def architecture(self) -> str:
        return self.gguf.metadata.get("general.architecture", "unknown")
        
    @property
    def total_size_bytes(self) -> int:
        return self.gguf.total_tensor_bytes
        
    @property
    def tensors(self) -> list:
        return self.gguf.tensors
        
    def summary(self) -> dict:
        return self.gguf.summary()


class ModelLoader:
    """
    Automatically detects model format and uses the correct parser.
    Currently supports:
    - GGUF (.gguf)
    - Safetensors (.safetensors) [Placeholder for future implementation]
    """
    
    @classmethod
    def load(cls, file_path: str | Path) -> UniversalModel:
        """
        Detect file type and load the model.
        Returns a UniversalModel conforming object.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
            
        suffix = path.suffix.lower()
        
        if suffix == ".gguf" or cls._detect_gguf(path):
            logger.info(f"Detected GGUF format for {path}")
            parser = GGUFParser(path)
            gguf_file = parser.parse()
            return GGUFUniversalAdapter(gguf_file, parser)
            
        elif suffix == ".safetensors":
            logger.info(f"Detected Safetensors format for {path}")
            return cls._load_safetensors(path)
            
        else:
            raise ValueError(f"Unsupported model format for file: {path}")
            
    @staticmethod
    def _detect_gguf(path: Path) -> bool:
        """Check the first 4 bytes for GGUF magic (both endians)."""
        try:
            with open(path, "rb") as f:
                magic = f.read(4)
                return magic in (b"GGUF", b"FUGG")
        except Exception:
            return False
            
    @staticmethod
    def _load_safetensors(path: Path) -> UniversalModel:
        """Placeholder for safetensors support using huggingface_hub/safetensors."""
        raise NotImplementedError(
            "Safetensors support is currently being implemented. "
            "Please use GGUF format for now."
        )
