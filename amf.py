"""
AMF (Atomic Model Fragmentation) Library
=========================================
Public API for interacting with the AMF system as a Python library.

Usage:
    import amf
    
    # Load any model
    model = amf.load_universal("path/to/model.gguf")
    
    # Fragment the model into functional cells
    cells = amf.fragment(model, strategy="functional", output_dir="./cells")
"""

import logging
from pathlib import Path
from typing import Optional, Union

from core.model_loader import ModelLoader, UniversalModel
from core.sorting_algorithm import SortingAlgorithm, FragmentationStrategy
from core.cell_taxonomy import CellManifest

logger = logging.getLogger(__name__)

# Re-export key components for library users
from core.model_loader import ModelLoader
from engine.molecular_engine import MolecularEngine


def load_universal(model_path: Union[str, Path]) -> UniversalModel:
    """
    Load a model file securely, automatically detecting its format
    (GGUF, Safetensors, etc.).
    
    Args:
        model_path: Path to the model file.
        
    Returns:
        A loaded UniversalModel ready for fragmentation.
    """
    return ModelLoader.load(model_path)


def fragment(
    model: UniversalModel,
    output_dir: Union[str, Path] = "./cells",
    strategy: str = "functional",
    **kwargs
) -> CellManifest:
    """
    Fragment a loaded model into functional semantic cells.
    
    Args:
        model: A loaded UniversalModel instance.
        output_dir: Directory to save the generated .cell files and manifest.
        strategy: Fragmentation strategy ('functional', 'per_layer', 'hybrid', 'per_component')
        
    Returns:
        The generated CellManifest containing metadata about all cells.
    """
    strategy_map = {
        "functional": FragmentationStrategy.FUNCTIONAL,
        "per_layer": FragmentationStrategy.PER_LAYER,
        "per_component": FragmentationStrategy.PER_COMPONENT,
        "hybrid": FragmentationStrategy.HYBRID,
    }
    
    str_enum = strategy_map.get(strategy.lower(), FragmentationStrategy.FUNCTIONAL)
    
    # Currently, our sorting algorithm expects a GGUF file and its wrapper.
    # In a full universal implementation, the sorting algorithm would operate
    # purely on the UniversalModel interface.
    from core.model_loader import GGUFUniversalAdapter
    
    if not isinstance(model, GGUFUniversalAdapter):
        raise NotImplementedError(
            f"Fragmentation for {type(model).__name__} is not yet supported. "
            "Only GGUF models are currently fully supported for fragmentation."
        )
    
    logger.info(f"Starting fragmentation using {strategy} strategy...")
    
    sorter = SortingAlgorithm(
        gguf_path=model.gguf.path,
        output_dir=output_dir,
        strategy=str_enum,
        **kwargs
    )
    
    # Since we already parsed the model, we can bypass the parsing step in sorting
    sorter.gguf_file = model.gguf
    sorter.parser = model.parser
    
    manifest = sorter.execute()
    return manifest
