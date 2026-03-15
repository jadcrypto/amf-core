"""
Terminal Interface
==================
Rich-powered terminal UI for the AMF system.
Provides an interactive chat experience with real-time
cell loading visualization and system statistics.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Optional

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.layout import Layout
    from rich.live import Live
    from rich.markdown import Markdown
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

logger = logging.getLogger(__name__)


class TerminalUI:
    """
    Interactive terminal interface for the AMF system.

    Commands:
        /stats  - Show engine statistics
        /cells  - Show loaded cells
        /memory - Show memory usage
        /help   - Show help
        /quit   - Exit
    """

    BANNER = r"""
    ╔══════════════════════════════════════════════════════════════╗
    ║         ⚛️  Atomic Model Fragmentation (AMF) System         ║
    ║         ─────────────────────────────────────────           ║
    ║              المحرك الجزيئي — Molecular Engine              ║
    ╚══════════════════════════════════════════════════════════════╝
    """

    COMMANDS = {
        "/stats": "Show engine statistics",
        "/cells": "Show loaded cells and their DNA tags",
        "/memory": "Show memory usage breakdown",
        "/manifest": "Show cell manifest summary",
        "/help": "Show this help message",
        "/quit": "Exit the application",
        "/clear": "Clear the screen",
    }

    def __init__(self, engine=None, inference_engine=None):
        self.engine = engine
        self.inference_engine = inference_engine
        self.console = Console() if HAS_RICH else None
        self._running = False

    def run(self):
        """Start the interactive chat loop."""
        self._running = True
        self._show_banner()

        if self.engine:
            self._show_init_status()

        self._print_info("Type /help for commands. Type your message to chat.\n")

        while self._running:
            try:
                prompt = self._get_input()
                if prompt is None:
                    break
                self._handle_input(prompt)
            except KeyboardInterrupt:
                self._print_info("\n\nInterrupted. Type /quit to exit.")
            except Exception as e:
                self._print_error(f"Error: {e}")
                logger.exception("Error in terminal loop")

        self._print_info("\n👋 مع السلامة — Goodbye!\n")

    def _handle_input(self, text: str):
        """Process user input — either a command or a chat message."""
        text = text.strip()
        if not text:
            return

        # Handle commands
        if text.startswith("/"):
            self._handle_command(text)
            return

        # Chat message — process through engine
        self._handle_chat(text)

    def _handle_command(self, cmd: str):
        """Process a slash command."""
        cmd = cmd.lower().strip()

        if cmd == "/quit" or cmd == "/exit":
            self._running = False
        elif cmd == "/help":
            self._show_help()
        elif cmd == "/stats":
            self._show_stats()
        elif cmd == "/cells":
            self._show_cells()
        elif cmd == "/memory":
            self._show_memory()
        elif cmd == "/manifest":
            self._show_manifest()
        elif cmd == "/clear":
            if self.console:
                self.console.clear()
            else:
                print("\033[2J\033[H")
        else:
            self._print_warning(f"Unknown command: {cmd}. Type /help for options.")

    def _handle_chat(self, prompt: str):
        """Process a chat message through the engine."""
        if not self.engine:
            self._print_error("Engine not initialized.")
            return

        self._print_info("⚡ Analyzing intent and loading cells...")

        # Process through molecular engine
        result = self.engine.process(prompt)

        # Show cell loading info
        intent = result["intent"]
        self._print_cell_info(
            f"🧬 Intent: {intent.primary_intent} "
            f"(confidence: {intent.confidence:.0%})"
        )
        if intent.is_compound:
            self._print_cell_info(
                f"   Compound: {', '.join(intent.compound_intents)}"
            )
        self._print_cell_info(
            f"📦 Loaded cells: {', '.join(result['loaded_cells'])}"
        )
        self._print_cell_info(
            f"🧩 Tensors: {result['n_tensors']} | "
            f"Memory: {result['memory_used_mb']:.1f} MB | "
            f"Time: {result['elapsed_ms']:.1f} ms"
        )

        # Generate response if inference engine is available
        if self.inference_engine:
            self._print_info("🤔 Generating response...")
            try:
                response = self.inference_engine.generate(
                    prompt, max_tokens=128, temperature=0.7
                )
                self._print_response(response)
            except Exception as e:
                self._print_error(f"Inference error: {e}")
                logger.exception("Inference error")
        else:
            self._print_warning(
                "Inference engine not available. "
                "Showing cell loading results only."
            )

    # ====================================================
    # Display methods
    # ====================================================

    def _show_banner(self):
        if self.console and HAS_RICH:
            self.console.print(
                Panel(
                    Text.from_markup(
                        "[bold cyan]⚛️  Atomic Model Fragmentation (AMF) System[/]\n"
                        "[dim]المحرك الجزيئي — Molecular Engine[/]"
                    ),
                    box=box.DOUBLE_EDGE,
                    style="bright_blue",
                    padding=(1, 4),
                )
            )
        else:
            print(self.BANNER)

    def _show_init_status(self):
        if not self.engine:
            return
        stats = self.engine.get_stats()
        manifest = stats.get("manifest", {})
        loader = stats.get("loader", {})
        self._print_info(
            f"📊 Model: {manifest.get('model', 'N/A')} | "
            f"Cells: {manifest.get('total_cells', 0)} | "
            f"Core loaded: {loader.get('loaded_cells', 0)} cells "
            f"({loader.get('used_memory_mb', 0):.1f} MB)"
        )

    def _show_help(self):
        if self.console and HAS_RICH:
            table = Table(title="Commands", box=box.ROUNDED)
            table.add_column("Command", style="cyan bold")
            table.add_column("Description")
            for cmd, desc in self.COMMANDS.items():
                table.add_row(cmd, desc)
            self.console.print(table)
        else:
            print("\nCommands:")
            for cmd, desc in self.COMMANDS.items():
                print(f"  {cmd:12s} {desc}")
            print()

    def _show_stats(self):
        if not self.engine:
            self._print_error("Engine not initialized")
            return

        stats = self.engine.get_stats()
        engine_stats = stats.get("engine", {})

        if self.console and HAS_RICH:
            table = Table(title="Engine Statistics", box=box.ROUNDED)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            for key, value in engine_stats.items():
                table.add_row(key, str(value))
            self.console.print(table)
        else:
            print("\n--- Engine Statistics ---")
            for key, value in engine_stats.items():
                print(f"  {key}: {value}")

    def _show_cells(self):
        if not self.engine or not self.engine.cell_loader:
            self._print_error("Engine not initialized")
            return

        stats = self.engine.cell_loader.get_stats()
        cells = stats.get("cells", {})

        if self.console and HAS_RICH:
            table = Table(title="Loaded Cells", box=box.ROUNDED)
            table.add_column("Cell ID", style="cyan bold")
            table.add_column("DNA Tag", style="magenta")
            table.add_column("Size (MB)", style="green")
            table.add_column("Ref Count", style="yellow")
            table.add_column("Tensors", style="blue")

            for cell_id, info in cells.items():
                table.add_row(
                    cell_id,
                    info["dna_tag"],
                    str(info["size_mb"]),
                    str(info["ref_count"]),
                    str(info["n_tensors"]),
                )
            self.console.print(table)
        else:
            print("\n--- Loaded Cells ---")
            for cell_id, info in cells.items():
                print(
                    f"  {cell_id:20s} | {info['dna_tag']:15s} | "
                    f"{info['size_mb']:6.2f} MB | ref={info['ref_count']}"
                )

    def _show_memory(self):
        if not self.engine or not self.engine.cell_loader:
            self._print_error("Engine not initialized")
            return

        stats = self.engine.cell_loader.get_stats()

        used = stats["used_memory_mb"]
        total = stats["max_memory_mb"]
        available = stats["available_memory_mb"]
        pct = (used / total * 100) if total > 0 else 0

        if self.console and HAS_RICH:
            # Memory bar
            bar_width = 40
            filled = int(bar_width * pct / 100)
            bar = "█" * filled + "░" * (bar_width - filled)
            color = "green" if pct < 60 else "yellow" if pct < 80 else "red"

            self.console.print(
                Panel(
                    f"[{color}]{bar}[/] {pct:.1f}%\n"
                    f"Used: {used:.1f} MB / {total:.0f} MB\n"
                    f"Available: {available:.1f} MB\n"
                    f"Loaded cells: {stats['loaded_cells']}",
                    title="Memory Usage",
                    box=box.ROUNDED,
                )
            )
        else:
            print(f"\n--- Memory Usage ---")
            print(f"  Used:      {used:.1f} MB / {total:.0f} MB ({pct:.1f}%)")
            print(f"  Available: {available:.1f} MB")
            print(f"  Cells:     {stats['loaded_cells']}")

    def _show_manifest(self):
        if not self.engine or not self.engine.manifest:
            self._print_error("Manifest not loaded")
            return

        summary = self.engine.manifest.summary()

        if self.console and HAS_RICH:
            table = Table(title="Cell Manifest", box=box.ROUNDED)
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")
            for key, value in summary.items():
                table.add_row(key, str(value))
            self.console.print(table)
        else:
            print("\n--- Cell Manifest ---")
            for key, value in summary.items():
                print(f"  {key}: {value}")

    # ====================================================
    # I/O helpers
    # ====================================================

    def _get_input(self) -> Optional[str]:
        try:
            if self.console and HAS_RICH:
                return self.console.input("[bold cyan]You >[/] ")
            else:
                return input("You > ")
        except EOFError:
            return None

    def _print_response(self, text: str):
        if self.console and HAS_RICH:
            self.console.print(
                Panel(
                    text,
                    title="[bold green]AMF Response[/]",
                    border_style="green",
                    box=box.ROUNDED,
                    padding=(0, 1),
                )
            )
        else:
            print(f"\nAMF > {text}\n")

    def _print_info(self, text: str):
        if self.console and HAS_RICH:
            self.console.print(f"[dim]{text}[/]")
        else:
            print(text)

    def _print_cell_info(self, text: str):
        if self.console and HAS_RICH:
            self.console.print(f"[bright_magenta]{text}[/]")
        else:
            print(text)

    def _print_warning(self, text: str):
        if self.console and HAS_RICH:
            self.console.print(f"[yellow]⚠️  {text}[/]")
        else:
            print(f"WARNING: {text}")

    def _print_error(self, text: str):
        if self.console and HAS_RICH:
            self.console.print(f"[bold red]❌ {text}[/]")
        else:
            print(f"ERROR: {text}")
