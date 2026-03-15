"""
AMF System — Main Entry Point
==============================
Atomic Model Fragmentation system.

Usage:
    # Fragment a model (first time)
    python main.py fragment --model qwen2.5:0.5b

    # Run interactive chat
    python main.py chat

    # Show system info
    python main.py info
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    PROJECT_ROOT, CELLS_DIR, MANIFEST_FILE,
    MODEL_CONFIG, TARGET_MODEL,
    MAX_RAM_BUDGET_MB,
)


def setup_logging(level: str = "INFO"):
    """Configure logging."""
    log_format = (
        "%(asctime)s | %(levelname)-7s | %(name)-25s | %(message)s"
    )
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=log_format,
        datefmt="%H:%M:%S",
    )


def cmd_fragment(args):
    """Fragment a model into cells."""
    from core.gguf_parser import find_gguf_model
    from core.sorting_algorithm import SortingAlgorithm, FragmentationStrategy

    print("=" * 60)
    print("⚛️  AMF — Model Fragmentation")
    print("=" * 60)

    # Locate the model file
    model_path = None
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        print(f"🔍 Searching for model: {args.model}...")
        model_path = find_gguf_model(args.model)

    if model_path is None or not model_path.exists():
        print(f"❌ Model not found: {args.model}")
        print(f"   Searched: Ollama default paths and {PROJECT_ROOT / 'models'}")
        print(f"\n   To fix, either:")
        print(f"   1. Run: ollama pull {args.model}")
        print(f"   2. Place a .gguf file in: {PROJECT_ROOT / 'models'}/")
        print(f"   3. Specify path: python main.py fragment --model-path /path/to/model.gguf")
        return 1

    print(f"📦 Model file: {model_path}")
    print(f"📂 Output dir: {CELLS_DIR}")
    print(f"🔧 Strategy: {args.strategy}")
    print()

    # Select strategy
    strategy_map = {
        "functional": FragmentationStrategy.FUNCTIONAL,
        "per_layer": FragmentationStrategy.PER_LAYER,
        "per_component": FragmentationStrategy.PER_COMPONENT,
        "hybrid": FragmentationStrategy.HYBRID,
    }
    strategy = strategy_map.get(args.strategy, FragmentationStrategy.FUNCTIONAL)

    # Run fragmentation
    sorter = SortingAlgorithm(
        gguf_path=model_path,
        output_dir=CELLS_DIR,
        strategy=strategy,
        n_layers=MODEL_CONFIG["n_layers"],
    )

    try:
        manifest = sorter.execute()
        print("\n✅ Fragmentation complete!")
        print(f"   Cells: {manifest.total_cells}")
        print(f"   Total size: {manifest.total_bytes / 1024 / 1024:.2f} MB")
        print(f"   Manifest: {MANIFEST_FILE}")

        # Save tokenizer separately for faster loading
        from engine.tokenizer import Tokenizer
        tokenizer = Tokenizer()
        tokenizer.load_from_gguf_metadata(sorter.gguf_file.metadata)
        tokenizer_path = CELLS_DIR / "tokenizer.json"
        tokenizer.save(tokenizer_path)
        print(f"   Tokenizer: {tokenizer_path}")

    except Exception as e:
        print(f"❌ Fragmentation failed: {e}")
        logging.exception("Fragmentation error")
        return 1
    finally:
        sorter.cleanup()

    return 0


def cmd_chat(args):
    """Run interactive chat."""
    from engine.molecular_engine import MolecularEngine
    from engine.inference import InferenceEngine
    from engine.tokenizer import Tokenizer
    from cli.terminal import TerminalUI

    # Check if cells exist
    if not MANIFEST_FILE.exists():
        print("❌ No fragmented cells found!")
        print("   Run first: python main.py fragment")
        return 1

    print("🔄 Initializing AMF system...\n")

    # Initialize molecular engine
    engine = MolecularEngine(
        cells_dir=CELLS_DIR,
        manifest_path=MANIFEST_FILE,
        max_memory_mb=MAX_RAM_BUDGET_MB,
    )
    engine.initialize()

    # Initialize tokenizer
    tokenizer = Tokenizer()
    tokenizer_path = CELLS_DIR / "tokenizer.json"
    if tokenizer_path.exists():
        tokenizer.load(tokenizer_path)
    else:
        print("⚠️  Tokenizer not found. Run fragment first.")
        return 1

    # Initialize inference engine
    inference = InferenceEngine(
        cell_loader=engine.cell_loader,
        tokenizer=tokenizer,
        n_layers=MODEL_CONFIG["n_layers"],
        n_embd=MODEL_CONFIG["n_embd"],
        n_head=MODEL_CONFIG["n_head"],
        n_head_kv=MODEL_CONFIG["n_head_kv"],
        n_ff=MODEL_CONFIG["n_ff"],
        rope_theta=MODEL_CONFIG["rope_theta"],
        rms_norm_eps=MODEL_CONFIG["rms_norm_eps"],
    )

    # Run terminal UI
    ui = TerminalUI(engine=engine, inference_engine=inference)

    try:
        ui.run()
    finally:
        engine.shutdown()

    return 0


def cmd_info(args):
    """Show system information."""
    print("=" * 60)
    print("⚛️  AMF System Information")
    print("=" * 60)
    print(f"\n📁 Project root: {PROJECT_ROOT}")
    print(f"📦 Cells dir:    {CELLS_DIR}")
    print(f"🧠 Target model: {TARGET_MODEL}")
    print(f"\n⚙️  Model Configuration:")
    for key, value in MODEL_CONFIG.items():
        print(f"   {key:20s}: {value}")
    print(f"\n💾 Memory Budget: {MAX_RAM_BUDGET_MB} MB")

    # Check if cells exist
    if MANIFEST_FILE.exists():
        from core.cell_taxonomy import CellManifest
        manifest = CellManifest.load(MANIFEST_FILE)
        summary = manifest.summary()
        print(f"\n📊 Cell Manifest:")
        for key, value in summary.items():
            print(f"   {key:20s}: {value}")
    else:
        print(f"\n⚠️  No cells found. Run: python main.py fragment")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="⚛️ AMF — Atomic Model Fragmentation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py fragment              # Fragment default model\n"
            "  python main.py fragment --strategy hybrid\n"
            "  python main.py chat                   # Interactive chat\n"
            "  python main.py info                   # System info\n"
        ),
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Fragment command
    frag_parser = subparsers.add_parser(
        "fragment", help="Fragment a model into cells"
    )
    frag_parser.add_argument(
        "--model", default=TARGET_MODEL,
        help=f"Model name (default: {TARGET_MODEL})"
    )
    frag_parser.add_argument(
        "--model-path", default=None,
        help="Direct path to a .gguf file (overrides --model)"
    )
    frag_parser.add_argument(
        "--strategy", default="functional",
        choices=["functional", "per_layer", "per_component", "hybrid"],
        help="Fragmentation strategy (default: functional)"
    )

    # Chat command
    subparsers.add_parser("chat", help="Interactive chat")

    # Info command
    subparsers.add_parser("info", help="Show system information")

    args = parser.parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level)

    if args.command is None:
        parser.print_help()
        return 0

    # Dispatch
    commands = {
        "fragment": cmd_fragment,
        "chat": cmd_chat,
        "info": cmd_info,
    }
    handler = commands.get(args.command)
    if handler:
        return handler(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main() or 0)
