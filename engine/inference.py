"""
Inference Engine
================
Performs forward pass inference using selectively loaded weight cells.

Implements the core Transformer computations:
- RMSNorm normalization
- Rotary Position Embeddings (RoPE)
- Multi-Head Attention with GQA (Grouped Query Attention)
- SwiGLU Feed-Forward Network
- Token embedding and output projection

All operations use NumPy for portability and full control
over selective weight activation.
"""

import logging
import math
import time
from typing import Optional

import numpy as np

from engine.cell_loader import CellLoader
from engine.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """RMSNorm: x * weight / sqrt(mean(x^2) + eps)"""
    variance = np.mean(x ** 2, axis=-1, keepdims=True)
    x_norm = x / np.sqrt(variance + eps)
    return x_norm * weight


def rope_frequencies(dim: int, seq_len: int, theta: float = 1000000.0) -> tuple:
    """Compute rotary position embedding frequencies."""
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    positions = np.arange(seq_len, dtype=np.float32)
    angles = np.outer(positions, freqs)
    cos_vals = np.cos(angles)
    sin_vals = np.sin(angles)
    return cos_vals, sin_vals


def apply_rope(x: np.ndarray, cos_vals: np.ndarray, sin_vals: np.ndarray) -> np.ndarray:
    """Apply Rotary Position Embeddings to query/key tensors."""
    # x shape: (seq_len, n_heads, head_dim)
    head_dim = x.shape[-1]
    half = head_dim // 2

    x1 = x[..., :half]
    x2 = x[..., half:]

    # Reshape cos/sin to broadcast
    cos_v = cos_vals[:x.shape[0], :half]
    sin_v = sin_vals[:x.shape[0], :half]

    # Add head dimension for broadcasting
    if x.ndim == 3:
        cos_v = cos_v[:, np.newaxis, :]
        sin_v = sin_v[:, np.newaxis, :]

    rotated = np.concatenate([
        x1 * cos_v - x2 * sin_v,
        x2 * cos_v + x1 * sin_v,
    ], axis=-1)
    return rotated


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU activation: x * sigmoid(x)"""
    return x * (1.0 / (1.0 + np.exp(-x)))


def dequantize_q8_0(raw: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Dequantize Q8_0 quantized data.
    Block format: 2 bytes scale (f16) + 32 bytes data (int8)
    """
    block_size = 32
    bytes_per_block = 34  # 2 (scale) + 32 (data)
    n_elements = 1
    for d in shape:
        n_elements *= d
    n_blocks = n_elements // block_size

    result = np.zeros(n_elements, dtype=np.float32)

    for i in range(n_blocks):
        offset = i * bytes_per_block
        # Read f16 scale
        scale = np.frombuffer(raw[offset:offset + 2], dtype=np.float16).astype(np.float32)[0]
        # Read int8 data
        data = np.frombuffer(raw[offset + 2:offset + bytes_per_block], dtype=np.int8).astype(np.float32)
        result[i * block_size:(i + 1) * block_size] = data * scale

    return result.reshape(shape)


def get_tensor_f32(cell_loader: CellLoader, tensor_name: str) -> Optional[np.ndarray]:
    """
    Retrieve a tensor from loaded cells and convert to float32.
    Handles dequantization for quantized tensors.
    """
    for cell_id, cell in cell_loader._loaded.items():
        for tv in cell.tensors:
            if tv.name == tensor_name:
                data = cell.get_tensor_data(tensor_name)
                if data is None:
                    return None
                if tv.dtype_name == "F32":
                    return data
                elif tv.dtype_name == "F16":
                    return data.astype(np.float32)
                elif tv.dtype_name == "Q8_0":
                    return dequantize_q8_0(data, tv.shape)
                else:
                    # For other quant types, return as-is
                    # Full dequantization support can be added per type
                    logger.warning(
                        f"Unsupported quant type {tv.dtype_name} for {tensor_name}, "
                        "returning raw bytes"
                    )
                    return data
    return None


class InferenceEngine:
    """
    Performs inference using selectively loaded weight cells.

    Architecture (Qwen2.5):
    - Embedding → [N × (Attention + FFN)] → RMSNorm → Output
    - Attention: QKV projection → GQA → Output projection
    - FFN: Gate/Up (SwiGLU) → Down projection
    """

    def __init__(
        self,
        cell_loader: CellLoader,
        tokenizer: Tokenizer,
        n_layers: int = 24,
        n_embd: int = 896,
        n_head: int = 14,
        n_head_kv: int = 2,
        n_ff: int = 4864,
        rope_theta: float = 1000000.0,
        rms_norm_eps: float = 1e-6,
    ):
        self.cell_loader = cell_loader
        self.tokenizer = tokenizer
        self.n_layers = n_layers
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_head_kv = n_head_kv
        self.n_ff = n_ff
        self.head_dim = n_embd // n_head
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps

        # KV cache for autoregressive generation
        self._kv_cache = {}

        # Pre-compute RoPE frequencies
        self._rope_cos = None
        self._rope_sin = None

    def _ensure_rope(self, max_seq_len: int):
        """Ensure RoPE frequencies are computed."""
        if self._rope_cos is None or self._rope_cos.shape[0] < max_seq_len:
            self._rope_cos, self._rope_sin = rope_frequencies(
                self.head_dim, max_seq_len, self.rope_theta
            )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy, >0 = random)
            top_p: Nucleus sampling parameter

        Returns:
            Generated text string
        """
        logger.info(f"Generating (max_tokens={max_tokens}, temp={temperature})")
        start_time = time.perf_counter()

        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, add_bos=True)
        logger.info(f"Prompt: {len(input_ids)} tokens")

        # Pre-compute RoPE for max possible length
        self._ensure_rope(len(input_ids) + max_tokens)

        # Clear KV cache
        self._kv_cache = {}

        generated = list(input_ids)

        for step in range(max_tokens):
            # Forward pass on last token (with KV cache)
            logits = self._forward(generated)

            # Sample next token
            next_token = self._sample(logits, temperature, top_p)

            # Check for EOS
            if next_token == self.tokenizer.eos_token_id:
                logger.info(f"EOS at step {step}")
                break

            generated.append(next_token)

            if step % 10 == 0:
                logger.debug(f"Step {step}/{max_tokens}")

        elapsed = time.perf_counter() - start_time
        n_generated = len(generated) - len(input_ids)
        tps = n_generated / elapsed if elapsed > 0 else 0

        logger.info(
            f"Generated {n_generated} tokens in {elapsed:.2f}s "
            f"({tps:.1f} tokens/sec)"
        )

        # Decode output (skip prompt tokens)
        output = self.tokenizer.decode(generated[len(input_ids):])
        return output

    def _forward(self, token_ids: list[int]) -> np.ndarray:
        """
        Forward pass through the transformer.

        Returns logits for the last token position.
        """
        seq_len = len(token_ids)

        # Token embedding
        embd_weight = get_tensor_f32(self.cell_loader, "token_embd.weight")
        if embd_weight is None:
            raise RuntimeError("Token embedding not loaded!")

        # Look up embeddings for all tokens
        # embd_weight shape: (n_vocab, n_embd)
        hidden = embd_weight[token_ids]  # (seq_len, n_embd)

        # Process through transformer layers
        for layer_idx in range(self.n_layers):
            hidden = self._transformer_block(hidden, layer_idx, seq_len)

        # Final RMSNorm
        norm_weight = get_tensor_f32(self.cell_loader, "output_norm.weight")
        if norm_weight is not None:
            hidden = rms_norm(hidden, norm_weight, self.rms_norm_eps)

        # Output projection — only compute for last token
        last_hidden = hidden[-1:]  # (1, n_embd)

        output_weight = get_tensor_f32(self.cell_loader, "output.weight")
        if output_weight is None:
            # Try shared embedding weights
            output_weight = embd_weight

        logits = last_hidden @ output_weight.T  # (1, n_vocab)
        return logits[0]  # (n_vocab,)

    def _transformer_block(
        self,
        hidden: np.ndarray,
        layer_idx: int,
        seq_len: int,
    ) -> np.ndarray:
        """Single transformer block: Attention + FFN with residual connections."""

        # --- Attention sub-block ---
        attn_norm_w = get_tensor_f32(
            self.cell_loader, f"blk.{layer_idx}.attn_norm.weight"
        )
        if attn_norm_w is not None:
            normed = rms_norm(hidden, attn_norm_w, self.rms_norm_eps)
            attn_out = self._attention(normed, layer_idx, seq_len)
            if attn_out is not None:
                hidden = hidden + attn_out

        # --- FFN sub-block ---
        ffn_norm_w = get_tensor_f32(
            self.cell_loader, f"blk.{layer_idx}.ffn_norm.weight"
        )
        if ffn_norm_w is not None:
            normed = rms_norm(hidden, ffn_norm_w, self.rms_norm_eps)
            ffn_out = self._ffn(normed, layer_idx)
            if ffn_out is not None:
                hidden = hidden + ffn_out

        return hidden

    def _attention(
        self,
        x: np.ndarray,
        layer_idx: int,
        seq_len: int,
    ) -> Optional[np.ndarray]:
        """
        Multi-Head Attention with Grouped Query Attention (GQA).

        Qwen2.5 0.5b: 14 heads, 2 KV heads → 7 Q heads share each KV head.
        """
        prefix = f"blk.{layer_idx}"

        q_weight = get_tensor_f32(self.cell_loader, f"{prefix}.attn_q.weight")
        k_weight = get_tensor_f32(self.cell_loader, f"{prefix}.attn_k.weight")
        v_weight = get_tensor_f32(self.cell_loader, f"{prefix}.attn_v.weight")
        o_weight = get_tensor_f32(self.cell_loader, f"{prefix}.attn_output.weight")

        if any(w is None for w in [q_weight, k_weight, v_weight, o_weight]):
            logger.debug(f"Skipping attention for layer {layer_idx} (tensors not loaded)")
            return None

        # Project to Q, K, V
        q = x @ q_weight.T  # (seq, n_head * head_dim)
        k = x @ k_weight.T  # (seq, n_head_kv * head_dim)
        v = x @ v_weight.T  # (seq, n_head_kv * head_dim)

        # Check for attention biases
        q_bias = get_tensor_f32(self.cell_loader, f"{prefix}.attn_q.bias")
        k_bias = get_tensor_f32(self.cell_loader, f"{prefix}.attn_k.bias")
        v_bias = get_tensor_f32(self.cell_loader, f"{prefix}.attn_v.bias")
        if q_bias is not None:
            q = q + q_bias
        if k_bias is not None:
            k = k + k_bias
        if v_bias is not None:
            v = v + v_bias

        # Reshape to (seq, n_heads, head_dim)
        q = q.reshape(seq_len, self.n_head, self.head_dim)
        k = k.reshape(seq_len, self.n_head_kv, self.head_dim)
        v = v.reshape(seq_len, self.n_head_kv, self.head_dim)

        # Apply RoPE
        q = apply_rope(q, self._rope_cos, self._rope_sin)
        k = apply_rope(k, self._rope_cos, self._rope_sin)

        # GQA: repeat K/V heads to match Q heads
        n_rep = self.n_head // self.n_head_kv
        if n_rep > 1:
            k = np.repeat(k, n_rep, axis=1)
            v = np.repeat(v, n_rep, axis=1)

        # Scaled dot-product attention
        # q, k, v: (seq, n_head, head_dim)
        scale = 1.0 / math.sqrt(self.head_dim)

        # Transpose to (n_head, seq, head_dim) for batch matmul
        q = q.transpose(1, 0, 2)
        k = k.transpose(1, 0, 2)
        v = v.transpose(1, 0, 2)

        # Attention scores: (n_head, seq, seq)
        attn_scores = np.matmul(q, k.transpose(0, 2, 1)) * scale

        # Causal mask
        mask = np.triu(np.full((seq_len, seq_len), -np.inf), k=1)
        attn_scores = attn_scores + mask

        attn_weights = softmax(attn_scores, axis=-1)

        # Attention output: (n_head, seq, head_dim)
        attn_out = np.matmul(attn_weights, v)

        # Reshape back: (seq, n_head * head_dim)
        attn_out = attn_out.transpose(1, 0, 2).reshape(seq_len, -1)

        # Output projection
        result = attn_out @ o_weight.T
        return result

    def _ffn(
        self,
        x: np.ndarray,
        layer_idx: int,
    ) -> Optional[np.ndarray]:
        """
        SwiGLU Feed-Forward Network.

        SwiGLU: output = Down(SiLU(Gate(x)) * Up(x))
        """
        prefix = f"blk.{layer_idx}"

        gate_w = get_tensor_f32(self.cell_loader, f"{prefix}.ffn_gate.weight")
        up_w = get_tensor_f32(self.cell_loader, f"{prefix}.ffn_up.weight")
        down_w = get_tensor_f32(self.cell_loader, f"{prefix}.ffn_down.weight")

        if any(w is None for w in [gate_w, up_w, down_w]):
            logger.debug(f"Skipping FFN for layer {layer_idx} (tensors not loaded)")
            return None

        # SwiGLU
        gate = silu(x @ gate_w.T)
        up = x @ up_w.T
        hidden = gate * up

        # Down projection
        output = hidden @ down_w.T
        return output

    def _sample(
        self,
        logits: np.ndarray,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> int:
        """Sample next token from logits using temperature + top-p."""
        if temperature <= 0:
            # Greedy
            return int(np.argmax(logits))

        # Temperature scaling
        logits = logits / temperature
        probs = softmax(logits)

        # Top-p (nucleus) sampling
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumsum = np.cumsum(sorted_probs)
        cutoff_idx = np.searchsorted(cumsum, top_p) + 1
        top_indices = sorted_indices[:cutoff_idx]
        top_probs = sorted_probs[:cutoff_idx]

        # Re-normalize
        top_probs = top_probs / top_probs.sum()

        # Sample
        chosen = np.random.choice(top_indices, p=top_probs)
        return int(chosen)
