"""
Tokenizer
=========
Extracts and wraps the tokenizer from the GGUF model metadata.
Handles encoding (text → token IDs) and decoding (token IDs → text).

For Qwen2.5, the tokenizer is BPE-based with a large vocabulary (~151K tokens).
The tokenizer data is embedded in the GGUF file's metadata section.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class Tokenizer:
    """
    Simple tokenizer that uses vocabulary extracted from GGUF metadata.

    For the initial version, we use a basic lookup-based approach.
    The full BPE merging algorithm can be added later for better quality.
    """

    def __init__(self):
        self.vocab: list[str] = []        # Index → token string
        self.token_to_id: dict = {}        # Token string → index
        self.n_vocab: int = 0
        self.bos_token_id: int = 0
        self.eos_token_id: int = 0
        self.pad_token_id: int = 0
        self._merges: list = []
        self._initialized = False

    def load_from_gguf_metadata(self, metadata: dict):
        """
        Extract tokenizer data from GGUF file metadata.

        Expected metadata keys:
        - tokenizer.ggml.model: "gpt2" (BPE type)
        - tokenizer.ggml.tokens: list of token strings
        - tokenizer.ggml.token_type: list of token types
        - tokenizer.ggml.merges: BPE merge rules
        - tokenizer.ggml.bos_token_id: beginning of sequence token
        - tokenizer.ggml.eos_token_id: end of sequence token
        """
        logger.info("Loading tokenizer from GGUF metadata...")

        # Extract vocabulary
        tokens = metadata.get("tokenizer.ggml.tokens", [])
        if not tokens:
            raise ValueError("No tokenizer tokens found in GGUF metadata")

        self.vocab = list(tokens)
        self.n_vocab = len(self.vocab)
        self.token_to_id = {
            token: idx for idx, token in enumerate(self.vocab)
        }

        # Special tokens
        self.bos_token_id = metadata.get("tokenizer.ggml.bos_token_id", 0)
        self.eos_token_id = metadata.get("tokenizer.ggml.eos_token_id", 0)
        self.pad_token_id = metadata.get("tokenizer.ggml.padding_token_id", 0)

        # BPE merges
        merges = metadata.get("tokenizer.ggml.merges", [])
        self._merges = merges

        self._initialized = True
        logger.info(
            f"Tokenizer loaded: {self.n_vocab} tokens, "
            f"{len(self._merges)} merges, "
            f"BOS={self.bos_token_id}, EOS={self.eos_token_id}"
        )

    def encode(self, text: str, add_bos: bool = True) -> list[int]:
        """
        Encode text into token IDs.

        Uses a simple greedy longest-match approach.
        For production use, this should implement proper BPE.
        """
        if not self._initialized:
            raise RuntimeError("Tokenizer not initialized")

        tokens = []
        if add_bos:
            tokens.append(self.bos_token_id)

        # Simple byte-level encoding as fallback
        text_bytes = text.encode("utf-8")

        i = 0
        while i < len(text_bytes):
            # Try longest match first
            best_match = None
            best_length = 0

            # Try various lengths (longest first for greedy matching)
            for length in range(min(20, len(text_bytes) - i), 0, -1):
                candidate = text_bytes[i:i + length]
                try:
                    candidate_str = candidate.decode("utf-8", errors="strict")
                except UnicodeDecodeError:
                    continue

                if candidate_str in self.token_to_id:
                    best_match = candidate_str
                    best_length = length
                    break

            if best_match is not None:
                tokens.append(self.token_to_id[best_match])
                i += best_length
            else:
                # Fallback: encode single byte as hex token
                byte_token = f"<0x{text_bytes[i]:02X}>"
                if byte_token in self.token_to_id:
                    tokens.append(self.token_to_id[byte_token])
                else:
                    # Unknown token — use a placeholder
                    tokens.append(0)
                i += 1

        return tokens

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs back to text."""
        if not self._initialized:
            raise RuntimeError("Tokenizer not initialized")

        parts = []
        for tid in token_ids:
            if tid == self.bos_token_id or tid == self.eos_token_id:
                continue
            if 0 <= tid < self.n_vocab:
                token = self.vocab[tid]
                # Handle byte tokens like <0xFF>
                if token.startswith("<0x") and token.endswith(">"):
                    try:
                        byte_val = int(token[3:-1], 16)
                        parts.append(bytes([byte_val]))
                        continue
                    except ValueError:
                        pass
                parts.append(token.encode("utf-8", errors="replace"))
            else:
                parts.append(b"?")

        # Join bytes and decode
        result = b"".join(parts)
        return result.decode("utf-8", errors="replace")

    def save(self, path: Path):
        """Save tokenizer data to a JSON file."""
        data = {
            "vocab": self.vocab,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
            "n_vocab": self.n_vocab,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        logger.info(f"Tokenizer saved to {path}")

    def load(self, path: Path):
        """Load tokenizer data from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab = data["vocab"]
        self.token_to_id = {t: i for i, t in enumerate(self.vocab)}
        self.bos_token_id = data["bos_token_id"]
        self.eos_token_id = data["eos_token_id"]
        self.pad_token_id = data["pad_token_id"]
        self.n_vocab = data["n_vocab"]
        self._initialized = True
        logger.info(f"Tokenizer loaded from {path}: {self.n_vocab} tokens")
