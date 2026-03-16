"""
amf — Atomic Model Fragmentation
=================================
First Arabic-originated Python library providing a
Molecular Inference Engine for large language models.

Run any GGUF model with < 500 MB RAM.

Quick start
-----------
>>> import amf
>>>
>>> # Direct inference (fastest path)
>>> engine = amf.engine("path/to/model.gguf")
>>> engine.load()
>>> print(engine.predict("Hello"))        # e.g. "resilient"
>>> print(engine.generate("Hello", n=5)) # "Hello resilient strong bold clear"
>>> engine.close()
>>>
>>> # Full fragmentation pipeline
>>> model = amf.load_universal("path/to/model.gguf")
>>> cells = amf.fragment(model, strategy="functional", output_dir="./cells")

Public API
----------
engine(model_path, ...)   → AMFEngine   (molecular inference)
load_universal(path)      → UniversalModel
fragment(model, ...)      → CellManifest
AMFEngine                 (class — direct import)
MolecularEngine           (class — full pipeline)
"""

import logging
from pathlib import Path
from typing import Union

# ── Core imports ────────────────────────────────────────────────────────────
from engine.amf_engine     import AMFEngine
from engine.molecular_engine import MolecularEngine
from core.model_loader     import ModelLoader, UniversalModel
from core.sorting_algorithm import SortingAlgorithm, FragmentationStrategy
from core.cell_taxonomy    import CellManifest

logger = logging.getLogger(__name__)

__version__     = "0.2.0"
__author__      = "Jad"
__license__     = "MIT"
__description__ = (
    "Atomic Model Fragmentation (AMF) — "
    "Molecular Inference Engine for resource-constrained hardware."
)
__all__ = [
    "engine",
    "load_universal",
    "fragment",
    "AMFEngine",
    "MolecularEngine",
]


# ── Public factory functions ─────────────────────────────────────────────────

def engine(
    model_path: Union[str, Path],
    inference_layer: int = 20,
    max_vocab_scan: int  = 100_000,
) -> AMFEngine:
    """
    Create an AMFEngine for direct molecular inference.

    Parameters
    ----------
    model_path : str | Path
        Path to a GGUF model file.
    inference_layer : int
        Transformer block index used for deep inference (default: 20).
    max_vocab_scan : int
        Maximum vocabulary tokens to score (default: 100 000).

    Returns
    -------
    AMFEngine
        Call ``.load()`` before using.

    Examples
    --------
    >>> eng = amf.engine("qwen2.5-7b.gguf")
    >>> eng.load()
    >>> eng.predict("Hello")
    'resilient'
    """
    return AMFEngine(
        model_path      = model_path,
        inference_layer = inference_layer,
        max_vocab_scan  = max_vocab_scan,
    )


def load_universal(model_path: Union[str, Path]) -> UniversalModel:
    """
    Load a model file, auto-detecting its format (GGUF, Safetensors …).

    Parameters
    ----------
    model_path : str | Path
        Path to the model file.

    Returns
    -------
    UniversalModel
        A loaded model ready for fragmentation.
    """
    return ModelLoader.load(model_path)


def fragment(
    model: UniversalModel,
    output_dir: Union[str, Path] = "./cells",
    strategy: str = "functional",
    **kwargs,
) -> CellManifest:
    """
    Fragment a loaded model into independent semantic cells.

    Parameters
    ----------
    model : UniversalModel
        A model loaded via :func:`load_universal`.
    output_dir : str | Path
        Directory where cell files and the manifest are saved.
    strategy : str
        Fragmentation strategy:

        * ``"functional"`` *(default)* — group by function
        * ``"per_layer"``              — one cell per transformer layer
        * ``"hybrid"``                 — mix of both
        * ``"per_component"``          — one cell per weight component

    Returns
    -------
    CellManifest
        Manifest containing metadata for all generated cells.

    Examples
    --------
    >>> model = amf.load_universal("model.gguf")
    >>> cells = amf.fragment(model, strategy="functional", output_dir="./cells")
    >>> print(f"Generated {cells.total_cells} cells")
    """
    _strategy_map = {
        "functional":   FragmentationStrategy.FUNCTIONAL,
        "per_layer":    FragmentationStrategy.PER_LAYER,
        "per_component":FragmentationStrategy.PER_COMPONENT,
        "hybrid":       FragmentationStrategy.HYBRID,
    }
    strat = _strategy_map.get(strategy.lower(), FragmentationStrategy.FUNCTIONAL)

    from core.model_loader import GGUFUniversalAdapter
    if not isinstance(model, GGUFUniversalAdapter):
        raise NotImplementedError(
            f"Fragmentation for {type(model).__name__} is not yet supported. "
            "Only GGUF models are fully supported."
        )

    logger.info(f"Fragmenting model | strategy={strategy}")
    sorter            = SortingAlgorithm(
        gguf_path  = model.gguf.path,
        output_dir = output_dir,
        strategy   = strat,
        **kwargs,
    )
    sorter.gguf_file  = model.gguf
    sorter.parser     = model.parser
    return sorter.execute()
