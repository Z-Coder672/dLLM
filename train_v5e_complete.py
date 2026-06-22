#!/usr/bin/env python3
"""
Complete v5e TPU training script with full training loop implementation.

Features:
- JAX/Flax model with proper Transformer architecture
- Full cross-entropy loss computation
- Google Drive checkpoint integration
- Gradient clipping and mixed precision (BF16)
- Log-spaced checkpoint pruning
- Indefinite training with manual stop (SIGINT/SIGTERM checkpoint-then-exit)

Usage:
    python train_v5e_complete.py --config configs/v5e.yaml
    python train_v5e_complete.py --config configs/v5e.yaml --auto-resume
"""

import argparse
import time
import math
import logging
import os
import json
import shutil
import signal
import sys
import threading
import queue
import random
from functools import partial
from pathlib import Path
from typing import Optional, Dict, Tuple, Any
from datetime import datetime

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, value_and_grad, lax
except ImportError:
    print("ERROR: JAX not installed. Install with: pip install 'jax[tpu]'")
    sys.exit(1)

import numpy as np
import yaml


# ============================================================================
# LOGGING & SETUP
# ============================================================================

logger = None


class LoggerWrapper:
    """Wrapper that both prints and logs messages."""
    def __init__(self, logger):
        self.logger = logger
    
    def info(self, msg):
        """Log and print info message."""
        print(msg)
        self.logger.info(msg)
    
    def debug(self, msg):
        """Log and print debug message."""
        print(msg)
        self.logger.debug(msg)

    def warning(self, msg):
        """Log and print warning message."""
        print(msg)
        self.logger.warning(msg)

    def error(self, msg):
        """Log and print error message."""
        print(msg)
        self.logger.error(msg)


def setup_logging(log_file: str = "training_v5e_complete.log"):
    """Setup logging to both stdout and a log file.

    The log file is opened in APPEND mode: with --auto-resume restart loops the
    previous run's log is exactly what's needed to diagnose why a run died, so
    it is never truncated. Each run is delimited by a timestamped banner.
    """
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    wrapper = LoggerWrapper(logger)
    wrapper.info("=" * 70)
    wrapper.info(
        f"=== New run started {datetime.now().isoformat(timespec='seconds')} ==="
    )
    wrapper.info("=" * 70)
    return wrapper


def setup_google_drive():
    """Mount Google Drive in Colab."""
    try:
        from google.colab import drive
        drive.mount('/content/gdrive', force_remount=False)
        logger.info("Google Drive mounted at /content/gdrive")
        return '/content/gdrive'
    except ImportError:
        logger.info("Not running in Colab, using local paths")
        return None
    except Exception as e:
        logger.warning(f"Failed to mount Google Drive: {e}")
        return None


def load_config(config_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load YAML config."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config.get('model', {}), config.get('training', {})


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

def rope_tables(
    seq_len: int, head_dim: int, theta: float, dtype: Any
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Build the (1, 1, seq_len, head_dim) cos/sin RoPE tables.

    These depend only on (seq_len, head_dim, theta) — constant across all layers —
    so the caller computes them ONCE per forward and threads them into every
    block, rather than rebuilding them 24x and relying on XLA to CSE the dup.

    Cast to the query dtype (bf16 in the forward pass) so the rotation product
    `q * cos_emb` stays bf16 instead of being promoted to f32 — an f32 table here
    silently drags the entire residual stream (and every downstream matmul, all
    layers) onto the f32 path, losing the MXU bf16 fast path while still paying
    the bf16 cast cost.
    """
    # Frequencies: (1/theta)^(2i/d) for i in [0, d/2)
    inv_freq = 1.0 / (theta ** (jnp.arange(0, head_dim, 2.0) / head_dim))
    t = jnp.arange(seq_len, dtype=jnp.float32)
    freqs = jnp.einsum("i,j->ij", t, inv_freq)
    emb = jnp.concatenate([freqs, freqs], axis=-1)
    cos_emb = jnp.cos(emb).astype(dtype)[None, None, :, :]
    sin_emb = jnp.sin(emb).astype(dtype)[None, None, :, :]
    return cos_emb, sin_emb


def apply_rope(
    q: jnp.ndarray, k: jnp.ndarray, cos_emb: jnp.ndarray, sin_emb: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Apply precomputed RoPE cos/sin tables (see ``rope_tables``) to q, k."""
    d = q.shape[-1]

    def rotate_half(x):
        x1, x2 = x[..., :d // 2], x[..., d // 2:]
        return jnp.concatenate([-x2, x1], axis=-1)

    q_rot = q * cos_emb + rotate_half(q) * sin_emb
    k_rot = k * cos_emb + rotate_half(k) * sin_emb
    return q_rot, k_rot


class TransformerModel:
    """500M Transformer model in JAX."""
    
    def __init__(self, model_config: Dict[str, Any], key: jax.random.PRNGKey):
        self.d_model = model_config['d_model']
        self.n_layers = model_config['n_layers']
        self.n_heads = model_config['n_heads']
        self.d_ff = model_config['d_ff']
        self.vocab_size = model_config['vocab_size']
        self.max_seq_len = model_config.get('max_seq_len', 512)
        self.rope_theta = model_config.get('rope_theta', 10000.0)
        self.head_dim = self.d_model // self.n_heads
        # Weight tying: when enabled, the output projection reuses the token
        # embedding (logits = x @ embed.T) instead of a separate lm_head matrix,
        # saving ~vocab*d_model params. Default off so untied checkpoints resume
        # unchanged; tying is a fresh-run choice (an untied checkpoint can't be
        # converted to tied without discarding a trained matrix).
        self.tie_embeddings = model_config.get('tie_embeddings', False)
        self.dtype = jnp.bfloat16
        # Row-chunk size for the memory-efficient cross-entropy (see compute_loss).
        # 0 disables chunking. Overridden from training config in `main`.
        self.ce_chunk_size = 0
        # Rematerialize transformer blocks in backward (jax.checkpoint): without
        # it autodiff stores every block's activations across all n_layers —
        # several GB at 500M scale, and the real cap on batch size once the CE
        # is chunked. Same memory/recompute lever as the chunked CE. Overridden
        # from training config in `main`.
        self.remat_blocks = True

        self.params = self._init_params(key)
    
    def _init_params(self, key: jax.random.PRNGKey) -> Dict[str, jnp.ndarray]:
        """Initialize all parameters."""
        params = {}
        
        # Token embedding
        key, subkey = jax.random.split(key)
        params['embed'] = jax.random.normal(
            subkey,
            (self.vocab_size, self.d_model),
            dtype=self.dtype
        ) * 0.02
        
        # Transformer layers
        for layer_idx in range(self.n_layers):
            params[f'layer_{layer_idx}'] = self._init_layer(key, layer_idx)
            key, _ = jax.random.split(key)
        
        # Final norm and output projection. When tying, the output head is
        # embed.T (added in forward), so no separate lm_head parameter exists.
        params['final_norm'] = jnp.ones(self.d_model, dtype=self.dtype)
        if not self.tie_embeddings:
            key, subkey = jax.random.split(key)
            params['lm_head'] = jax.random.normal(
                subkey,
                (self.d_model, self.vocab_size),
                dtype=self.dtype
            ) * 0.02

        return params
    
    def _init_layer(self, key: jax.random.PRNGKey, layer_idx: int) -> Dict[str, jnp.ndarray]:
        """Initialize single transformer layer."""
        layer = {}
        
        # Attention norms and projections
        layer['attn_norm'] = jnp.ones(self.d_model, dtype=self.dtype)
        
        key, *subkeys = jax.random.split(key, 7)
        scale = 1.0 / math.sqrt(self.d_model)
        # Projections that feed the residual stream (out_proj, fc2) get GPT-2-style
        # depth-scaled init 0.02/sqrt(2*n_layers). A flat 0.02 lets residual-stream
        # activations grow with depth early in training (a minor instability risk).
        residual_scale = 0.02 / math.sqrt(2 * self.n_layers)

        layer['q_proj'] = jax.random.normal(
            subkeys[0], (self.d_model, self.d_model), dtype=self.dtype) * scale
        layer['k_proj'] = jax.random.normal(
            subkeys[1], (self.d_model, self.d_model), dtype=self.dtype) * scale
        layer['v_proj'] = jax.random.normal(
            subkeys[2], (self.d_model, self.d_model), dtype=self.dtype) * scale
        layer['out_proj'] = jax.random.normal(
            subkeys[3], (self.d_model, self.d_model), dtype=self.dtype) * residual_scale

        # FFN norms and weights
        layer['ffn_norm'] = jnp.ones(self.d_model, dtype=self.dtype)
        layer['fc1'] = jax.random.normal(
            subkeys[4], (self.d_model, self.d_ff), dtype=self.dtype) * 0.02
        layer['fc2'] = jax.random.normal(
            subkeys[5], (self.d_ff, self.d_model), dtype=self.dtype) * residual_scale
        
        return layer
    
    def backbone(
        self,
        input_ids: jnp.ndarray,
        params: Dict[str, Any],
    ) -> jnp.ndarray:
        """Embeddings + transformer blocks + final norm.

        Returns the final-normed hidden state (batch, seq_len, d_model) *before*
        the output projection. Kept separate from ``forward`` so the loss can fold
        the (large) vocab projection into a chunked cross-entropy without ever
        materializing the full (batch*seq, vocab) logits tensor (see compute_loss).
        """
        # Embedding: (batch, seq_len, d_model)
        x = params['embed'][input_ids.astype(jnp.int32)]
        seq_len = x.shape[1]

        # Build the RoPE tables and causal mask ONCE (they're layer-invariant) and
        # thread them into every block instead of recomputing them per layer.
        cos_emb, sin_emb = rope_tables(
            seq_len, self.head_dim, self.rope_theta, x.dtype
        )
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))

        # Process through transformer blocks. With remat_blocks, each block is
        # rematerialized: backward recomputes the block's forward instead of
        # keeping its activations live across all n_layers (see __init__).
        block_fn = (
            jax.checkpoint(self._transformer_block)
            if self.remat_blocks
            else self._transformer_block
        )
        for layer_idx in range(self.n_layers):
            x = block_fn(
                x, params[f'layer_{layer_idx}'], cos_emb, sin_emb, causal_mask
            )

        # Final norm: (batch, seq_len, d_model). Reduce in f32 — a bf16 mean over
        # d_model loses precision (same reason softmax/loss run in f32).
        xf = x.astype(jnp.float32)
        rms = jnp.sqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + 1e-6)
        return ((xf / rms).astype(x.dtype)) * params['final_norm']

    def output_weight(self, params: Dict[str, Any]) -> jnp.ndarray:
        """Output projection W, with logits = hidden @ W, shape (d_model, vocab).

        Weight tying reuses the token embedding transposed; otherwise the
        dedicated ``lm_head`` matrix.
        """
        return params['embed'].T if self.tie_embeddings else params['lm_head']

    def forward(
        self,
        input_ids: jnp.ndarray,
        params: Optional[Dict[str, Any]] = None,
        training: bool = True,
    ) -> jnp.ndarray:
        """Full forward pass returning logits (batch, seq_len, vocab_size).

        Training computes the loss via ``backbone`` + a chunked cross-entropy
        (compute_loss) instead; this full-logits path is kept for inference / any
        non-loss use that genuinely needs the materialized logits.

        ``training`` is accepted for API symmetry but unused: this model has no
        dropout (intentional for large-data pretraining), so train and eval
        forwards are identical. Kept as a parameter so callers don't break.
        """
        params_tree = params if params is not None else self.params
        x = self.backbone(input_ids, params_tree)
        # f32 logits via preferred_element_type (not a post-hoc .astype, which
        # would round to bf16 first); see _chunk_ce for the rationale.
        return jnp.dot(
            x, self.output_weight(params_tree), preferred_element_type=jnp.float32
        )
    
    def _transformer_block(
        self,
        x: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
        cos_emb: jnp.ndarray,
        sin_emb: jnp.ndarray,
        causal_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """Single transformer block with pre-norm.

        ``cos_emb``/``sin_emb`` (RoPE tables) and ``causal_mask`` are precomputed
        once in ``backbone`` and passed in, since they're identical across layers.
        """
        batch_size, seq_len, d_model = x.shape
        
        # Attention block
        # Normalize (reduce in f32; bf16 mean over d_model loses precision).
        xf = x.astype(jnp.float32)
        rms = jnp.sqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + 1e-6)
        x_norm = ((xf / rms).astype(x.dtype)) * params['attn_norm']
        
        # Project to Q, K, V
        q = jnp.dot(x_norm, params['q_proj'])  # (batch, seq, d_model)
        k = jnp.dot(x_norm, params['k_proj'])
        v = jnp.dot(x_norm, params['v_proj'])
        
        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Apply RoPE (tables precomputed once in backbone)
        q, k = apply_rope(q, k, cos_emb, sin_emb)
        
        # Attention scores. Feed bf16 q/k into the MXU but accumulate in f32
        # (preferred_element_type) so the 512-class softmax keeps f32 stability
        # without forcing the operands — and the residual stream — back to f32.
        scores = jnp.einsum(
            "bhqd,bhkd->bhqk", q, k, preferred_element_type=jnp.float32
        ) / math.sqrt(self.head_dim)

        # Causal mask (precomputed once in backbone)
        scores = jnp.where(causal_mask[None, None, :, :], scores, -1e9)
        
        # Attention weights and context. softmax runs in f32 for stability, but
        # cast the weights back to v's dtype before the context matmul so it (and
        # the residual stream below) stays on the bf16 MXU path.
        attn_weights = jax.nn.softmax(scores, axis=-1).astype(v.dtype)
        context = jnp.matmul(attn_weights, v)  # (batch, n_heads, seq, head_dim)
        
        # Merge heads
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        
        # Output projection
        attn_out = jnp.dot(context, params['out_proj'])
        x = x + attn_out
        
        # FFN block
        # Normalize (reduce in f32; bf16 mean over d_model loses precision).
        xf = x.astype(jnp.float32)
        rms = jnp.sqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + 1e-6)
        x_norm = ((xf / rms).astype(x.dtype)) * params['ffn_norm']
        
        # FFN: (batch, seq, d_ff) -> (batch, seq, d_model)
        ffn_out = jnp.dot(x_norm, params['fc1'])
        ffn_out = jax.nn.gelu(ffn_out)  # GELU activation (plain 2-matrix FFN, not SwiGLU)
        ffn_out = jnp.dot(ffn_out, params['fc2'])
        
        x = x + ffn_out
        
        return x
    
    def count_parameters(self) -> int:
        """Count total parameters."""
        def count_tree(tree):
            if isinstance(tree, jnp.ndarray):
                return np.prod(tree.shape)
            elif isinstance(tree, dict):
                return sum(count_tree(v) for v in tree.values())
            return 0
        return count_tree(self.params)


# ============================================================================
# OPTIMIZER & LOSS
# ============================================================================

class AdamWOptimizer:
    """AdamW optimizer."""
    
    def __init__(
        self,
        beta1: float = 0.9,
        beta2: float = 0.98,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        # No stored learning rate: the per-step LR (schedule + resume ramp) is
        # passed into `apply` as a traced value.
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = {}
        self.v = {}
        self.t = 0

    def init_state(self, params: Dict[str, Any]):
        """Initialize optimizer state tree."""
        def init_tree(tree):
            if isinstance(tree, jnp.ndarray):
                return jnp.zeros_like(tree, dtype=jnp.float32)
            elif isinstance(tree, dict):
                return {k: init_tree(v) for k, v in tree.items()}
            return tree
        
        self.m = init_tree(params)
        self.v = init_tree(params)
    
    @staticmethod
    def apply(
        params: Dict[str, Any],
        grads: Dict[str, Any],
        m: Dict[str, Any],
        v: Dict[str, Any],
        t,
        lr,
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float,
        decay_mask: Optional[Dict[str, Any]] = None,
    ):
        """
        Pure AdamW update over an entire pytree.

        This contains NO Python-level per-leaf dispatch at run time: it is meant
        to be wrapped in jax.jit (or inlined inside a larger jitted step) so XLA
        fuses the moment updates for every parameter tensor into a single program
        — one kernel launch instead of ~150. The Python `for` loop below runs only
        once, during tracing, to build that single graph.

        `t` and `lr` are passed as (traced) values rather than closed-over Python
        constants so the step count / LR schedule never trigger recompilation.

        Returns (new_params, new_m, new_v) as pytrees matching `params`.
        """
        bias_correction1 = 1.0 - beta1 ** t
        bias_correction2 = 1.0 - beta2 ** t

        p_leaves, treedef = jax.tree_util.tree_flatten(params)
        g_leaves = treedef.flatten_up_to(grads)
        m_leaves = treedef.flatten_up_to(m)
        v_leaves = treedef.flatten_up_to(v)
        # Per-leaf weight-decay multiplier (1.0 = decay, 0.0 = skip). Built by
        # `build_decay_mask` so 1-D RMSNorm gains (and optionally embeddings)
        # are excluded — decaying a norm gain toward 0 fights the normalization.
        # When absent, fall back to decaying every leaf (the historical behavior).
        if decay_mask is not None:
            mask_leaves = treedef.flatten_up_to(decay_mask)
        else:
            mask_leaves = [1.0] * len(p_leaves)

        new_p, new_m, new_v = [], [], []
        for p, g, m_leaf, v_leaf, decay_scale in zip(
            p_leaves, g_leaves, m_leaves, v_leaves, mask_leaves
        ):
            p_f32 = p.astype(jnp.float32)
            g_f32 = g.astype(jnp.float32)

            m_leaf = beta1 * m_leaf + (1.0 - beta1) * g_f32
            v_leaf = beta2 * v_leaf + (1.0 - beta2) * (g_f32 * g_f32)

            update = (m_leaf / bias_correction1) / (
                jnp.sqrt(v_leaf / bias_correction2) + eps
            )
            # `decay_scale` is a static Python 0.0/1.0, so XLA folds the term away
            # entirely for excluded leaves (no extra HBM traffic).
            if weight_decay > 0 and decay_scale:
                update = update + weight_decay * p_f32

            new_p.append((p_f32 - lr * update).astype(p.dtype))
            new_m.append(m_leaf)
            new_v.append(v_leaf)

        return (
            jax.tree_util.tree_unflatten(treedef, new_p),
            jax.tree_util.tree_unflatten(treedef, new_m),
            jax.tree_util.tree_unflatten(treedef, new_v),
        )

def build_decay_mask(params: Dict[str, Any], decay_embeddings: bool = True) -> Dict[str, Any]:
    """Build a per-leaf weight-decay mask mirroring the param tree.

    Each leaf becomes 1.0 (apply decay) or 0.0 (skip). 1-D RMSNorm gains
    (``attn_norm`` / ``ffn_norm`` / ``final_norm`` — every key ending in
    ``_norm``) are always excluded: decaying a normalization gain toward 0
    fights the norm it scales. The token ``embed`` is excluded when
    ``decay_embeddings=False``. The result is a dict with the same structure as
    ``params``, so it flattens in the same leaf order as ``AdamWOptimizer.apply``.
    """
    def walk(tree, key=""):
        if isinstance(tree, dict):
            return {k: walk(v, k) for k, v in tree.items()}
        if key.endswith("_norm"):
            return 0.0
        if key == "embed" and not decay_embeddings:
            return 0.0
        return 1.0

    return walk(params)


def _chunk_ce(x_chunk, labels_chunk, out_weight):
    """Summed cross-entropy for one row-chunk, folding the vocab projection in.

    Computes ``logits = x_chunk @ out_weight`` for *this chunk only* (so the full
    (batch*seq, vocab) tensor is never materialized) and returns the *summed*
    (not averaged) CE — callers divide by the global token count.
    ``CE = logsumexp(logits) - logits[label]`` avoids building a second full
    log_softmax array. Wrapped in ``jax.checkpoint`` by the caller so the chunk's
    logits are recomputed in backward instead of held in HBM.

    ``preferred_element_type=jnp.float32`` makes the bf16xbf16 MXU matmul emit f32
    logits *directly* — a plain ``jnp.dot(...).astype(f32)`` would round the
    product to bf16 first (only the internal accumulation is f32) and then upcast
    already-lost bits, leaving the 50k-class softmax running on bf16-resolution
    logits. The f32 output costs no extra HBM beyond the f32 logits we already
    need, and the bf16 operands keep the MXU fast path (same pattern as the
    attention-score matmul).
    """
    logits = jnp.dot(x_chunk, out_weight, preferred_element_type=jnp.float32)
    lse = jax.nn.logsumexp(logits, axis=-1)
    target = jnp.take_along_axis(
        logits, labels_chunk[:, None].astype(jnp.int32), axis=-1
    ).squeeze(-1)
    return jnp.sum(lse - target)


def compute_loss(
    params: Dict[str, Any],
    batch: Dict[str, jnp.ndarray],
    model: TransformerModel,
) -> jnp.ndarray:
    """Memory-efficient cross-entropy loss.

    The naive path materializes the full (batch*seq, vocab) logits in f32 *and* a
    second equally large log_softmax tensor — ~2 GB at batch 8 / seq 512 / 50k
    vocab, which is a big chunk of a 16 GB chip and the main cap on batch size.
    Instead, run the backbone to hidden states and fold the vocab projection into
    a chunked cross-entropy: each chunk's logits are computed, reduced
    (logsumexp - target logit), and discarded; the chunk fn is rematerialized so
    backward recomputes logits rather than storing them. Peak logit memory drops
    from O(batch*seq*vocab) to O(chunk*vocab). The result is numerically identical
    to a plain mean cross-entropy (chunking only reassociates an f32 sum).

    ``model.ce_chunk_size`` rows are processed per chunk; <=0 or >= total rows
    means a single chunk (still cheaper than the naive path — one big f32 array
    instead of two).
    """
    x = model.backbone(batch['input_ids'], params)
    out_weight = model.output_weight(params)

    _, _, d_model = x.shape
    x_flat = x.reshape(-1, d_model)
    labels_flat = batch['labels'].reshape(-1)
    n_rows = x_flat.shape[0]

    # Rematerialize the cross-entropy so the (rows, vocab) logits live only
    # transiently — backward recomputes them instead of holding them in HBM. This
    # also covers the single-chunk path below: WITHOUT the wrap, disabling
    # chunking (ce_chunk_size <= 0) would make autodiff store the full
    # (batch*seq, vocab) f32 logits for backward — the exact ~2 GB tensor chunking
    # exists to avoid — so "disable chunking" would paradoxically raise peak memory.
    ckpt_ce = jax.checkpoint(_chunk_ce)

    chunk = getattr(model, 'ce_chunk_size', 0) or n_rows
    if chunk <= 0 or chunk >= n_rows:
        return ckpt_ce(x_flat, labels_flat, out_weight) / n_rows

    total = jnp.float32(0.0)
    for start in range(0, n_rows, chunk):
        end = min(start + chunk, n_rows)
        total = total + ckpt_ce(
            x_flat[start:end], labels_flat[start:end], out_weight
        )
    return total / n_rows


def compute_perplexity(loss: float) -> float:
    """Compute perplexity, guarding against a non-finite loss.

    ``min(nan, 20)`` returns ``nan`` (Python ``min`` keeps the first arg when the
    comparison is False), so a NaN/Inf loss would silently surface as a bare
    ``nan``/``inf`` PPL. Return ``nan`` explicitly for any non-finite loss; for a
    finite loss clamp before ``exp`` so a large-but-finite loss doesn't overflow.
    """
    l = float(loss)
    if not math.isfinite(l):
        return float('nan')
    return math.exp(min(l, 20))


def clip_gradients(
    grads: Dict[str, Any],
    max_norm: float = 1.0,
) -> Tuple[Dict[str, Any], float]:
    """Clip gradients by global norm."""
    leaves = jax.tree_util.tree_leaves(grads)
    norm = jnp.sqrt(sum(jnp.sum(g * g) for g in leaves))
    scale = jnp.minimum(1.0, max_norm / (norm + 1e-8))
    return jax.tree_util.tree_map(lambda g: g * scale, grads), norm


# ============================================================================
# CHECKPOINTING
# ============================================================================

FLAT_KEY_SEP = "::"


def flatten_dict(d: Dict[str, Any], parent_key: str = '') -> Dict[str, np.ndarray]:
    """Flatten nested dictionary for saving."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{FLAT_KEY_SEP}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key).items())
        elif isinstance(v, jnp.ndarray):
            items.append((new_key, np.array(v)))
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: Dict[str, Any], dtype=None) -> Dict[str, Any]:
    """Unflatten dictionary after loading. If dtype is given, cast arrays to it."""
    result = {}
    for key, value in d.items():
        parts = key.split(FLAT_KEY_SEP)
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        if isinstance(value, np.ndarray):
            if value.dtype.kind == 'V' and value.dtype.itemsize == 2:
                value = np.frombuffer(value.tobytes(), dtype=np.uint16).reshape(value.shape)
                arr = jnp.array(value).view(jnp.bfloat16)
            else:
                arr = jnp.array(value)
            current[parts[-1]] = arr.astype(dtype) if dtype is not None else arr
        else:
            current[parts[-1]] = value
    return result


def _write_data_state(path, state):
    """Write a dataloader position snapshot to ``dataloader_state.json``.

    Best-effort: data-state IO must never break checkpointing, so any failure is
    logged and swallowed (a resume from this checkpoint then falls back to replay).
    """
    try:
        with open(Path(path) / "dataloader_state.json", "w") as f:
            json.dump(state, f)
    except Exception as e:
        logger.warning(
            f"Could not save dataloader state ({e}); a resume from this "
            "checkpoint will fall back to stream replay."
        )


# Checkpoint format version stamped into state.json. v2 introduced the staged
# (.tmp dir + rename) write plus a DONE completeness marker; list_checkpoints
# requires DONE for v2+ checkpoints but accepts marker-less legacy (v1) ones so
# existing checkpoints still resume.
CHECKPOINT_FORMAT = 2
CHECKPOINT_DONE_MARKER = "DONE"


def _remove_tree_drivesafe(path):
    """rmtree a checkpoint dir, truncating files to 0 bytes first.

    On Google Drive FUSE mounts shutil.rmtree sends files to Drive Trash, which
    still consumes quota; truncating first means even trashed files are empty.
    """
    for root, _dirs, files in os.walk(path):
        for fname in files:
            try:
                with open(os.path.join(root, fname), "wb") as f:
                    f.truncate(0)
            except OSError:
                pass
    shutil.rmtree(path)


def _fsync(path):
    """Best-effort flush of a file's or directory's bytes/metadata to disk.

    Pushes pending writes through the (FUSE) cache so a crash after this call
    can't leave the file silently empty/partial. Swallows errors — fsync may be
    unsupported on some mounts and durability is a best-effort hardening, not a
    correctness precondition (the DONE marker is the actual completeness gate).
    """
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except (OSError, ValueError):
        pass


def _checkpoint_payload(
    params: Dict[str, Any],
    optimizer: AdamWOptimizer,
    step: int,
    model_config: Dict[str, Any],
    training_config: Dict[str, Any],
    data_loader: Optional["StreamingDataLoader"] = None,
    data_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Pull everything a checkpoint needs OFF-DEVICE into host memory (MAIN thread).

    Split out from the disk write so the slow part (``np.savez_compressed`` over a
    Google Drive FUSE mount) can run on a background thread (see CheckpointSaver)
    while training continues. ``flatten_dict`` performs the device→host copy and
    ``int(optimizer.t)`` / ``get_learning_rate`` are read here, so the returned
    payload is a fully independent host snapshot — the next ``train_step`` may
    donate/overwrite the live device buffers without affecting it.

    ``data_state`` is the preferred data position: the per-batch snapshot from
    ``StreamingDataLoader.iter_with_state`` (matches the batch just trained on).
    Falls back to a live loader snapshot only when none is supplied, taken HERE on
    the main thread (snapshotting from the worker would race the advancing loader).
    """
    params_flat = flatten_dict(params)
    m_flat = flatten_dict(optimizer.m)
    v_flat = flatten_dict(optimizer.v)

    # Actual number of batches drawn from the deterministic stream so far. Equals
    # `step` only when gradient_accumulation_steps == 1; with accumulation each
    # optimizer step consumes accum_steps batches, so prefer the trained-batch
    # snapshot's own count, then the live loader, then `step`.
    if data_state is not None:
        data_batches = data_state.get("batches_consumed", step)
    elif data_loader is not None:
        data_batches = data_loader.batches_consumed
    else:
        data_batches = step
    # Live-loader fallback only when no per-batch snapshot was supplied.
    if data_state is None and data_loader is not None:
        data_state = data_loader.data_state()

    # Record the instantaneous LR (including any active resume-ramp) so a later
    # resume reconstructs it exactly. optimizer.t is a device scalar after step 1
    # (advanced inside train_step; see N3) — coerce to int for JSON here.
    state = {
        "format": CHECKPOINT_FORMAT,
        "step": step,
        "optimizer_t": int(optimizer.t),
        "learning_rate": get_learning_rate(step, training_config),
        "data_batches_consumed": data_batches,
    }
    # Strip the transient `lr_ramp_*` keys that `main` injects into the live
    # config at resume time. They describe a one-off ramp window for THIS run
    # only; persisting them would pollute the saved config and could skew the
    # old-checkpoint LR reconstruction path in load_checkpoint (M4).
    clean_training_config = {
        k: v for k, v in training_config.items() if not k.startswith("lr_ramp_")
    }
    return {
        "params_flat": params_flat,
        "m_flat": m_flat,
        "v_flat": v_flat,
        "state": state,
        "model_config": model_config,
        "training_config": clean_training_config,
        "data_state": data_state,
    }


def _write_checkpoint_payload(path: str, payload: Dict[str, Any]):
    """Write a host-memory checkpoint payload to disk, atomically.

    Everything is staged into a sibling ``<path>.tmp`` directory, every file is
    fsync'd, a ``DONE`` marker is written last, and only then is the staging dir
    published to the final path via ``os.replace``. The dominant failure — a
    crash during the multi-minute ``.npz`` writes over a Google Drive FUSE mount
    — therefore happens entirely under ``.tmp`` (which ``list_checkpoints``
    ignores), so the final path never exists half-written. The ``DONE`` marker is
    the authoritative completeness signal for the residual case where the publish
    rename itself is interrupted on a non-atomic FUSE rename.

    This is the SLOW half of a save and is what CheckpointSaver runs on a
    background thread; the payload is already host memory (see _checkpoint_payload).
    """
    final_path = Path(path)
    tmp_path = Path(str(final_path) + ".tmp")
    # Start from a clean staging dir (a leftover .tmp from a previously-killed
    # save would otherwise mix stale files into this checkpoint).
    if tmp_path.exists():
        _remove_tree_drivesafe(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(tmp_path / "params.npz", **payload["params_flat"])
    np.savez_compressed(tmp_path / "optimizer_m.npz", **payload["m_flat"])
    np.savez_compressed(tmp_path / "optimizer_v.npz", **payload["v_flat"])

    with open(tmp_path / "state.json", "w") as f:
        json.dump(payload["state"], f, indent=2)
    with open(tmp_path / "model_config.json", "w") as f:
        json.dump(payload["model_config"], f, indent=2)
    with open(tmp_path / "training_config.json", "w") as f:
        json.dump(payload["training_config"], f, indent=2)

    # Persist the exact data-stream position so a resume continues without
    # re-tokenizing the consumed prefix (best-effort; see StreamingDataLoader).
    if payload["data_state"] is not None:
        _write_data_state(tmp_path, payload["data_state"])

    # Durably flush every staged file BEFORE writing the DONE marker, so the
    # marker can never outlive the data it vouches for.
    for child in tmp_path.iterdir():
        _fsync(child)

    # DONE marker, written last and fsync'd: its presence (with state.json's
    # format>=2) is what list_checkpoints requires to accept the checkpoint.
    with open(tmp_path / CHECKPOINT_DONE_MARKER, "w") as f:
        f.write("ok\n")
        f.flush()
        os.fsync(f.fileno())

    # Publish atomically. Remove any existing final dir first (rerun / abort /
    # "final"); step-named checkpoints are unique so this is usually a no-op.
    if final_path.exists():
        _remove_tree_drivesafe(final_path)
    os.replace(tmp_path, final_path)
    _fsync(final_path.parent)  # record the rename durably in the parent directory

    logger.info(f"Saved checkpoint to {final_path}")


def save_checkpoint(
    path: str,
    params: Dict[str, Any],
    optimizer: AdamWOptimizer,
    step: int,
    model_config: Dict[str, Any],
    training_config: Dict[str, Any],
    data_loader: Optional["StreamingDataLoader"] = None,
    data_state: Optional[Dict[str, Any]] = None,
):
    """Save a checkpoint SYNCHRONOUSLY (host-copy + atomic disk write).

    Used by the abort / signal-stop / final-save paths, which must block until
    the bytes are durable (and, for the abort path, must finish before an
    uncatchable SIGKILL). The periodic in-loop saves instead go through
    ``_checkpoint_payload`` + ``CheckpointSaver`` so the TPU doesn't stall on the
    multi-minute Drive write; this wrapper is exactly that payload + write run
    back-to-back on the calling thread.
    """
    payload = _checkpoint_payload(
        params, optimizer, step, model_config, training_config,
        data_loader, data_state,
    )
    _write_checkpoint_payload(path, payload)


class CheckpointSaver:
    """Runs checkpoint disk-writes on a background thread.

    A save is dominated by ``np.savez_compressed`` of the ~5 GB f32 optimizer
    state over a Google Drive FUSE mount — minutes during which the TPU would
    otherwise sit idle. The device→host copy (``_checkpoint_payload``) is done on
    the MAIN thread before ``submit``, so the payload is an independent host
    snapshot; the worker compresses + writes it (and prunes) while training
    mutates the separate live device buffers.

    A maxsize-1 queue bounds in-flight work: with ``save_interval`` >> save
    duration only one save is ever in flight, so host memory holds ~one payload.
    ``flush`` blocks until pending writes finish — call it before any synchronous
    save and before process exit so a background write isn't lost or killed
    mid-flight.
    """

    def __init__(self):
        self._q: "queue.Queue" = queue.Queue(maxsize=1)
        self._thread = threading.Thread(
            target=self._worker, name="checkpoint-saver", daemon=True
        )
        self._thread.start()

    def _worker(self):
        while True:
            job = self._q.get()
            if job is None:
                self._q.task_done()
                break
            path, payload, prune_args = job
            try:
                _write_checkpoint_payload(path, payload)
                if prune_args is not None:
                    prune_checkpoints(*prune_args)
            except Exception as e:
                # A failed periodic checkpoint must not kill training — the next
                # one will save. Surface it loudly and carry on.
                logger.error(f"Async checkpoint save failed for {path}: {e}")
            finally:
                self._q.task_done()

    def submit(self, path, payload, prune_args=None):
        """Queue a host-memory payload for background write (+ optional prune).

        Blocks only if a previous save is still in flight (maxsize-1 backpressure),
        which under normal cadence never happens.
        """
        self._q.put((path, payload, prune_args))

    def flush(self):
        """Block until all queued saves (and prunes) have completed."""
        self._q.join()


def _log_config_drift(label, saved, current, log_fn):
    """Log every key whose value differs between a checkpoint's saved config
    and the live one. ``log_fn`` sets the severity per config kind."""
    diffs = [
        f"  {k}: checkpoint={saved.get(k)!r} -> current={current.get(k)!r}"
        for k in sorted(set(saved) | set(current))
        if saved.get(k) != current.get(k)
    ]
    if diffs:
        log_fn(
            f"{label} differs from this checkpoint ({len(diffs)} key(s)):\n"
            + "\n".join(diffs)
        )


def load_checkpoint(
    path: str,
    model: TransformerModel,
    optimizer: AdamWOptimizer,
    model_config: Optional[Dict[str, Any]] = None,
    training_config: Optional[Dict[str, Any]] = None,
) -> Tuple[int, Optional[float], int]:
    """Load checkpoint from directory.

    ``model_config``/``training_config`` (the live configs), when given, are
    diffed against the checkpoint's own saved configs and any drift is logged —
    model-config drift loudly (it silently corrupts: params are loaded
    wholesale, so e.g. a smaller ``n_layers`` orphans the extra layer trees,
    which still get weight-decayed and re-saved; a ``tie_embeddings`` flip
    KeyErrors at the first forward), training-config drift at info level (an
    enlarged ``max_steps`` is the documented resume workflow).

    Returns ``(start_step, old_lr, data_pos)`` where ``old_lr`` is the learning
    rate the checkpoint was actually being trained at — either read directly from
    ``state.json`` (newer checkpoints) or reconstructed from the checkpoint's
    own ``training_config.json`` + step (older checkpoints that predate LR
    persistence). ``old_lr`` is ``None`` only if it can't be determined.
    ``data_pos`` is the number of batches already consumed from the stream, used
    to fast-forward the data loader; it defaults to ``start_step`` for older
    checkpoints written before the field existed (the two are equal in practice).
    """
    path = Path(path)
    
    # Load params as the f32 master copy. Checkpoints written by this trainer
    # are already f32; older bf16 checkpoints simply upcast (no info change).
    # The forward pass downcasts to bf16 per step in `_loss_fn`.
    params_flat = dict(np.load(path / "params.npz"))
    model.params = unflatten_dict(params_flat, dtype=jnp.float32)
    
    # Load optimizer state (keep float32 for numerical stability)
    m_flat = dict(np.load(path / "optimizer_m.npz"))
    optimizer.m = unflatten_dict(m_flat)
    
    v_flat = dict(np.load(path / "optimizer_v.npz"))
    optimizer.v = unflatten_dict(v_flat)
    
    # Load training state
    with open(path / "state.json") as f:
        state = json.load(f)
    
    optimizer.t = state.get("optimizer_t", 0)
    start_step = state.get("step", 0)
    data_pos = state.get("data_batches_consumed", start_step)

    # Config-drift logs: surface any mismatch between the live configs and the
    # ones this checkpoint was trained with (see docstring for why model-config
    # drift is the dangerous one).
    if model_config is not None:
        try:
            with open(path / "model_config.json") as f:
                _log_config_drift("MODEL config", json.load(f), model_config,
                                  logger.warning)
        except (OSError, ValueError):
            pass
    if training_config is not None:
        try:
            with open(path / "training_config.json") as f:
                _log_config_drift("Training config", json.load(f),
                                  training_config, logger.info)
        except (OSError, ValueError):
            pass

    # Determine the LR the checkpoint was trained at (for smooth resume).
    old_lr = state.get("learning_rate")
    if old_lr is None:
        # Older checkpoint: reconstruct from its own training config + step.
        try:
            with open(path / "training_config.json") as f:
                old_training_config = json.load(f)
            old_lr = get_learning_rate(start_step, old_training_config)
        except (OSError, ValueError):
            old_lr = None

    lr_str = f"{old_lr:.3e}" if old_lr is not None else "unknown"
    logger.info(f"Loaded checkpoint from {path} at step {start_step} (trained LR ≈ {lr_str})")
    return start_step, old_lr, data_pos


def list_checkpoints(output_dir: str) -> list:
    """List all checkpoints."""
    path = Path(output_dir)
    if not path.exists():
        logger.info(f"Checkpoint dir does not exist: {path}")
        return []
    
    checkpoints = []
    for item in sorted(path.iterdir()):
        # Ignore in-progress staging dirs (a save_checkpoint that's still running
        # or was killed mid-write leaves its files under "<name>.tmp").
        if not item.is_dir() or item.name.endswith(".tmp"):
            continue
        if not (item / "state.json").exists():
            continue
        try:
            with open(item / "state.json") as f:
                state = json.load(f)
        except (OSError, ValueError) as e:
            # A corrupt/0-byte state.json must not brick every future
            # auto-resume. _remove_tree_drivesafe truncates files BEFORE
            # rmtree, so an interrupted prune/delete leaves exactly this
            # state behind — skip it, don't crash on it.
            logger.warning(f"Skipping unreadable checkpoint {item} ({e})")
            continue
        # v2+ checkpoints are only complete once the DONE marker (written last,
        # after every file is fsync'd) is present — this rejects a checkpoint whose
        # publish rename was interrupted on a non-atomic FUSE mount. Legacy v1
        # checkpoints had no marker, so accept them on state.json alone.
        if state.get("format", 1) >= 2 and not (item / CHECKPOINT_DONE_MARKER).exists():
            logger.warning(
                f"Skipping incomplete checkpoint (no DONE marker): {item}"
            )
            continue
        checkpoints.append({
            "path": str(item),
            "step": state.get("step", 0),
        })

    return sorted(checkpoints, key=lambda x: x["step"])


def get_latest_checkpoint(output_dir: str) -> Optional[str]:
    """Get latest checkpoint path."""
    checkpoints = list_checkpoints(output_dir)
    return checkpoints[-1]["path"] if checkpoints else None


def prune_checkpoints(output_dir: str, current_step: int, save_interval: int):
    """Prune old checkpoints using log-spaced thinning."""
    checkpoints = list_checkpoints(output_dir)
    if not checkpoints or save_interval <= 0:
        return
    
    keep_paths = set()
    bucket_best = {}
    
    for ckpt in checkpoints:
        step = ckpt["step"]
        path = ckpt["path"]
        age = max(0, current_step - step)
        
        if age < save_interval:
            bucket = 0
        else:
            bucket = int(math.floor(math.log(age / save_interval, 2.0)))
        best = bucket_best.get(bucket)
        
        if best is None or step > best[0]:
            bucket_best[bucket] = (step, path)
    
    # Always keep the latest checkpoint
    if checkpoints:
        latest = max(checkpoints, key=lambda c: c["step"])
        keep_paths.add(latest["path"])
    
    for _, path in bucket_best.values():
        keep_paths.add(path)
    
    for ckpt in checkpoints:
        path = ckpt["path"]
        # "final" marks a completed run — never prune it (a later resumed run's
        # buckets would otherwise happily discard it).
        if Path(path).name == "final":
            continue
        if path not in keep_paths:
            try:
                _remove_tree_drivesafe(path)
                logger.info(f"Pruned checkpoint: {path}")
            except Exception as e:
                logger.info(f"Failed to prune {path}: {e}")


def get_learning_rate(step: int, config: Dict[str, Any]) -> float:
    """Cosine annealing LR schedule, with an optional smooth ramp after a resume.

    When training is resumed under a changed schedule (e.g. ``max_steps`` was
    enlarged), the cosine curve generally jumps discontinuously at the resume
    point. To avoid shocking the optimizer we linearly interpolate from the LR
    the checkpoint was actually trained at (``lr_ramp_from``) to wherever the
    *new* curve will be ``lr_ramp_steps`` later (``lr_ramp_to``), over the window
    ``[lr_ramp_start_step, lr_ramp_start_step + lr_ramp_steps)``. Once the ramp
    finishes we simply follow the new cosine curve. These four ``lr_ramp_*`` keys
    are injected into the live config at resume time (see ``main``); when absent,
    this is a plain cosine schedule.
    """
    warmup_steps = config.get('warmup_steps', 2000)
    max_steps = config.get('max_steps', 1000000)
    min_lr = config.get('min_learning_rate', 1e-5)
    base_lr = config.get('learning_rate', 5e-4)

    def cosine_lr(s: float) -> float:
        if s < warmup_steps:
            return base_lr * (s / warmup_steps)
        # max(..., 1) guards max_steps == warmup_steps (smoke-test configs):
        # progress would otherwise divide by zero at the first post-warmup step.
        progress = (s - warmup_steps) / max(max_steps - warmup_steps, 1)
        progress = min(progress, 1.0)
        return min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

    ramp_start = config.get('lr_ramp_start_step')
    ramp_steps = config.get('lr_ramp_steps', 0)
    if ramp_start is not None and ramp_steps and ramp_start <= step < ramp_start + ramp_steps:
        ramp_from = config['lr_ramp_from']
        ramp_to = config['lr_ramp_to']
        frac = (step - ramp_start) / ramp_steps
        return ramp_from + (ramp_to - ramp_from) * frac

    return cosine_lr(step)


# ============================================================================
# JAX-COMPATIBLE DATA LOADING
# ============================================================================

def _get_tokenizer():
    """Get GPT-2 tokenizer."""
    import tiktoken
    return tiktoken.get_encoding("gpt2")


def _ensure_zstd():
    """Ensure zstandard is available (needed by some HF datasets)."""
    try:
        import zstandard  # noqa: F401
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "zstandard"])


def _load_hf_dataset(dataset_name, dataset_config=None, split="train",
                     shuffle_buffer=10000, seed=42, skip_first=0):
    """Load a single HuggingFace dataset with streaming.

    ``skip_first`` drops the first N raw examples *before* shuffling. This carves
    out a reserved-prefix eval holdout (N1): the training loader skips the first N
    raw examples of each stream while the eval set ``.take(N)`` reads exactly those,
    so the two are disjoint by construction — independent of the shuffle seed or
    how long training runs (a shuffled, position-based holdout can't guarantee
    that). Skip is applied to the raw stream so shuffling never pulls a held-out
    example into the training window.
    """
    _ensure_zstd()
    from datasets import DownloadConfig, load_dataset
    import os

    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_HTTP_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

    download_config = DownloadConfig(max_retries=5)

    kwargs = dict(split=split, streaming=True, download_config=download_config)
    if dataset_config:
        ds = load_dataset(dataset_name, dataset_config, **kwargs)
    else:
        ds = load_dataset(dataset_name, **kwargs)

    if skip_first > 0:
        ds = ds.skip(skip_first)
    if shuffle_buffer > 0:
        ds = ds.shuffle(buffer_size=shuffle_buffer, seed=seed)
    return ds


# Cache of (dataset name, config) -> whether a streaming "validation" split
# exists. Probed at most once per dataset and consulted by BOTH the training
# loader (whether to reserve a holdout prefix) and the eval-set builder (where
# eval reads from), so the two sides of the disjointness contract always make
# the same call.
_VALIDATION_SPLIT_CACHE: Dict[Tuple[str, Optional[str]], bool] = {}


def _has_validation_split(name, config=None):
    """True if the dataset exposes a (streaming) ``validation`` split. Cached."""
    cache_key = (name, config)
    if cache_key not in _VALIDATION_SPLIT_CACHE:
        try:
            _load_hf_dataset(name, config, "validation", shuffle_buffer=0)
            _VALIDATION_SPLIT_CACHE[cache_key] = True
        except Exception:
            _VALIDATION_SPLIT_CACHE[cache_key] = False
    return _VALIDATION_SPLIT_CACHE[cache_key]


# Datasets in the mixture store their document text under different keys
# (e.g. starcoderdata uses "content", not "text"). Try them in priority order.
_TEXT_FIELDS = ("text", "content", "code", "raw_content")

# Per-epoch shuffle-seed stride: when a finite stream is re-created on
# exhaustion, its seed becomes ``base + epoch * _EPOCH_SEED_STRIDE`` so each
# wraparound draws a different shuffled order. A large prime keeps per-epoch
# seeds well clear of adjacent streams' base seeds (42 + stream_index).
_EPOCH_SEED_STRIDE = 1_000_003


def _extract_text(example):
    """Return the first non-empty known text field from a streamed example."""
    for field in _TEXT_FIELDS:
        val = example.get(field)
        if val:
            return val
    return ""


def _packed_sequence_iter(dataset_iter, tokenizer, seq_len):
    """Pack tokenized text into fixed-length sequences (seq_len+1 for labels shift)."""
    eos = tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
    buf = []
    for example in dataset_iter:
        text = _extract_text(example)
        if not text:
            continue
        # encode_ordinary (not encode): the default encode disallows special
        # tokens and RAISES on any literal "<|endoftext|>" in the corpus, which
        # would crash validation-set construction at startup. This also matches
        # the training path (_encode_batch -> encode_ordinary_batch), so eval and
        # train tokenize identically.
        tokens = tokenizer.encode_ordinary(text)
        tokens.append(eos)
        buf.extend(tokens)
        while len(buf) >= seq_len + 1:
            yield buf[:seq_len + 1]
            buf = buf[seq_len:]


def prefetch(iterator, depth=3):
    """Run `iterator` on a background thread, buffering up to `depth` batches.

    The data pipeline tokenizes (tiktoken), builds NumPy arrays, and does the
    host→device transfer all in Python — synchronously between steps the TPU
    would idle through every one of those. Moving that work to a worker thread
    lets it overlap with on-device compute: while the TPU runs step N, the
    thread is already preparing step N+1 (and up to `depth` ahead). The GIL is
    released during tiktoken's native encode and during JAX's device transfer,
    so the overlap is real.

    A bounded queue provides backpressure (the producer blocks once `depth`
    batches are buffered) so an infinite training iterator can't run away and
    exhaust host memory. Exceptions raised in the producer are propagated to the
    consumer rather than silently killing the thread.
    """
    q = queue.Queue(maxsize=depth)
    _SENTINEL = object()

    def _producer():
        try:
            for item in iterator:
                q.put(item)
        except Exception as exc:  # surface producer errors on the main thread
            q.put(exc)
        finally:
            q.put(_SENTINEL)

    thread = threading.Thread(target=_producer, daemon=True)
    thread.start()

    while True:
        item = q.get()
        if item is _SENTINEL:
            break
        if isinstance(item, Exception):
            raise item
        yield item


def _encode_batch(tokenizer, texts):
    """Tokenize a list of documents, multithreaded when tiktoken supports it.

    ``encode_ordinary_batch`` releases the GIL and tokenizes the batch across
    native threads — markedly faster than per-document ``encode`` both in the
    steady state and when replaying the stream on a resume. ``encode_ordinary``
    (vs ``encode``) treats any literal ``<|endoftext|>`` in the corpus as ordinary
    text instead of raising; for all other text the two are identical, so the
    packed token stream matches what the per-document path produced (resume
    determinism against older checkpoints is preserved).
    """
    batch_fn = getattr(tokenizer, "encode_ordinary_batch", None)
    if batch_fn is not None:
        return batch_fn(texts)
    return [tokenizer.encode_ordinary(t) for t in texts]


class _PackedStream:
    """Packs ONE streaming dataset into fixed-length token sequences, resumably.

    Documents are pulled in batches and tokenized with ``_encode_batch`` (whole
    batches are folded into the token buffer before any sequence is yielded), so
    the pair ``(documents_consumed, leftover token buffer)`` fully describes the
    stream position. That makes an exact, tokenization-free resume possible:
    re-create the (deterministically shuffled) dataset, ``.skip(documents
    consumed)`` to the right document — streaming/parse only, no re-tokenization —
    and restore the leftover buffer.
    """

    def __init__(self, make_ds, tokenizer, seq_len, eos, doc_batch=128):
        self.make_ds = make_ds            # (epoch) -> fresh (shuffled) HF IterableDataset
        self.tok = tokenizer
        self.seq_len = seq_len
        self.eos = eos
        self.doc_batch = doc_batch
        # How many times this finite stream has been re-created from exhaustion.
        # Threaded into make_ds so each wraparound reshuffles with a different seed
        # (epoch 2 must not replay epoch 1's exact document order); persisted in the
        # position snapshot so a resume reproduces the same per-epoch shuffle.
        self.epoch = 0
        self.ds = make_ds(self.epoch)
        self._it = iter(self.ds)
        self.buf = []
        # Documents pulled from the stream since the current (re)load — i.e. the
        # `.skip()` offset that reproduces this position on a fresh load. Counts
        # every example pulled (including empty/text-less ones, which `.skip`
        # also passes over) so the offset stays exact.
        self.docs_consumed = 0

    def _fill(self):
        """Pull up to one doc-batch, fold all of it into the token buffer."""
        texts = []
        pulled = 0
        for _ in range(self.doc_batch):
            try:
                example = next(self._it)
            except StopIteration:
                break
            pulled += 1
            text = _extract_text(example)
            if text:
                texts.append(text)
        self.docs_consumed += pulled
        if texts:
            # Rebind self.buf to a NEW list rather than extending it in place.
            # A position snapshot (see _snapshot_locked) holds this list by
            # reference instead of deep-copying it every batch; an in-place
            # extend here would mutate an already-captured snapshot. next_seq
            # already reslices into a fresh list, so with this change self.buf is
            # only ever replaced, never mutated — captured references stay frozen.
            new_tokens = []
            for tokens in _encode_batch(self.tok, texts):
                new_tokens.extend(tokens)
                new_tokens.append(self.eos)
            self.buf = self.buf + new_tokens
        return pulled > 0

    def next_seq(self):
        """Return the next seq_len+1 token sequence, or raise StopIteration."""
        while len(self.buf) < self.seq_len + 1:
            if not self._fill():
                raise StopIteration
        seq = self.buf[:self.seq_len + 1]
        # 1-token overlap between consecutive sequences (next starts at this
        # sequence's last token) so no boundary token's prediction is dropped.
        self.buf = self.buf[self.seq_len:]
        return seq

    def restart(self):
        """Re-create the stream from the top (on natural exhaustion).

        Advances ``epoch`` so the fresh stream reshuffles with a new seed — a
        finite streaming dataset would otherwise replay its first epoch's exact
        document order on every wraparound, degrading data diversity on long runs.
        """
        self.epoch += 1
        self.ds = self.make_ds(self.epoch)
        self._it = iter(self.ds)
        self.buf = []
        self.docs_consumed = 0

    def resume(self, docs_consumed, buf, epoch=0):
        """Restore a saved position: fresh load, .skip() to it, restore buffer.

        ``epoch`` restores how many wraparounds had happened, so make_ds rebuilds
        the SAME per-epoch shuffle the checkpoint was on (older snapshots without
        the field default to 0 — the original behavior).

        The ``.skip`` is lazy — the actual stream/parse of skipped documents is
        paid on the first pull (which happens on the prefetch worker thread), and
        crucially involves no tokenization.
        """
        self.epoch = epoch
        self.ds = self.make_ds(self.epoch).skip(docs_consumed)
        self._it = iter(self.ds)
        self.buf = list(buf)
        self.docs_consumed = docs_consumed


class StreamingDataLoader:
    """Iterable, checkpoint-resumable loader: weighted dataset mixture -> batches.

    Yields dicts ``{'input_ids', 'labels'}`` (int32 JAX arrays, batch_size x
    seq_len). Iterating runs the mixture selector RNG over per-dataset
    ``_PackedStream``s. The whole position — selector RNG, each stream's
    ``(docs_consumed, buffer)``, and batch count — round-trips through
    ``data_state``/``load_data_state`` so a resume continues without replaying or
    re-tokenizing the consumed prefix. ``fast_forward`` remains as a fallback for
    checkpoints written before data-state was saved (it replays the stream, but
    now with multithreaded batch tokenization).
    """

    def __init__(self, datasets_config, dataset_name="wikitext",
                 dataset_config="wikitext-103-raw-v1", batch_size=8, seq_len=512,
                 shuffle_buffer=10000, seed=42, doc_batch=128, holdout_examples=0):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.shuffle_buffer = shuffle_buffer
        self.datasets_config = datasets_config
        # Reserved-prefix eval holdout (N1): every training stream skips its first
        # `holdout_examples` raw examples, which the eval set reads instead — so the
        # two are provably disjoint. Folded into the resume fingerprint so changing
        # it (or resuming a checkpoint that predates it) warns.
        self.holdout_examples = int(holdout_examples)
        self.is_mixture = bool(datasets_config)
        self.tokenizer = _get_tokenizer()
        self.eos = self.tokenizer.encode(
            "<|endoftext|>", allowed_special={"<|endoftext|>"}
        )[0]
        self.batches_consumed = 0
        # Selector RNG seeded so the mixture draw order is reproducible across
        # runs/resumes (an unseeded global RNG made data order non-deterministic).
        self._rng = random.Random(seed)
        # Guards stream/RNG mutation against state snapshots taken on the main
        # thread while the prefetch worker is producing batches.
        self._lock = threading.Lock()

        if self.is_mixture:
            total_w = sum(d.get("weight", 1.0) for d in datasets_config)
            c = 0.0
            self.cum_weights = []
            for d in datasets_config:
                c += d.get("weight", 1.0) / total_w
                self.cum_weights.append(c)
            self.streams = []
            self.stream_skips = []
            for i, cfg in enumerate(datasets_config):
                # Reserve the holdout prefix ONLY when this dataset has no real
                # validation split — eval then reads the prefix instead. When a
                # validation split exists, eval uses it and skipping the prefix
                # would just discard training data for nothing. The cached
                # probe is shared with _load_validation_split, so both sides of
                # the disjointness contract decide identically.
                skip = (0 if _has_validation_split(cfg["name"], cfg.get("config"))
                        else self.holdout_examples)
                if skip == 0 and self.holdout_examples:
                    logger.info(
                        f"Dataset '{cfg['name']}' has a real validation split; "
                        "no training holdout prefix reserved."
                    )
                self.stream_skips.append(skip)
                # `epoch` varies the shuffle seed on each wraparound so a finite
                # stream doesn't replay its first epoch's exact order. The stride
                # (_EPOCH_SEED_STRIDE) keeps per-epoch seeds from colliding with a
                # neighbour stream's base seed (42 + i).
                make_ds = (lambda epoch=0, cfg=cfg, i=i, skip=skip: _load_hf_dataset(
                    cfg["name"], cfg.get("config"), "train", shuffle_buffer,
                    seed=42 + i + epoch * _EPOCH_SEED_STRIDE,
                    skip_first=skip))
                self.streams.append(
                    _PackedStream(make_ds, self.tokenizer, seq_len, self.eos, doc_batch))
        else:
            skip = (0 if _has_validation_split(dataset_name, dataset_config)
                    else self.holdout_examples)
            if skip == 0 and self.holdout_examples:
                logger.info(
                    f"Dataset '{dataset_name}' has a real validation split; "
                    "no training holdout prefix reserved."
                )
            self.stream_skips = [skip]
            make_ds = (lambda epoch=0, skip=skip: _load_hf_dataset(
                dataset_name, dataset_config, "train", shuffle_buffer,
                seed=42 + epoch * _EPOCH_SEED_STRIDE,
                skip_first=skip))
            self.streams = [
                _PackedStream(make_ds, self.tokenizer, seq_len, self.eos, doc_batch)]
            self.cum_weights = [1.0]

        # Identity of the data source, captured ONCE (not per batch) for resume-
        # mismatch detection (N6); embedded by reference in every snapshot.
        self._fingerprint = self._compute_fingerprint()

    def _compute_fingerprint(self):
        """Identity of the data source, for resume-mismatch detection (N6).

        The no-retokenization resume relies on ``make_ds().skip(docs_consumed)``
        reproducing the SAME shuffled order as the original run. That holds only
        while the ``datasets`` library version, dataset revision, and mixture are
        unchanged. Record enough to warn loudly on load when they aren't (the
        ``.skip`` would otherwise land on different documents silently).
        """
        try:
            from importlib.metadata import version
            dv = version("datasets")
        except Exception:
            dv = "unknown"
        mixture = [
            {"name": d.get("name"), "config": d.get("config"),
             "weight": d.get("weight", 1.0)}
            for d in (self.datasets_config or [])
        ]
        # holdout_examples / per-stream skips shift the training stream
        # (skip_first), so a change since the checkpoint means .skip() lands on
        # different documents (N1/N6). stream_skips is the EFFECTIVE skip per
        # stream (0 when that dataset has a real validation split).
        return {"datasets_version": dv, "mixture": mixture,
                "holdout_examples": self.holdout_examples,
                "stream_skips": list(self.stream_skips)}

    def validate_streams(self, probe_batches=16):
        """Probe each mixture stream once; raise if any yields no usable data (N2).

        A dataset that produces nothing (wrong text field, missing config/
        ``data_dir``) is otherwise silently skipped at run time — ``_next_seq``
        warns and re-draws, redistributing that dataset's weight to the others and
        quietly changing the mixture. Catch it at startup instead. Uses fresh
        throwaway streams, so it never disturbs the live (possibly resumed) stream
        positions.
        """
        if not self.is_mixture:
            return
        for cfg, stream in zip(self.datasets_config, self.streams):
            probe = _PackedStream(
                stream.make_ds, self.tokenizer, self.seq_len, self.eos,
                stream.doc_batch,
            )
            produced = False
            for _ in range(probe_batches):
                if not probe._fill():            # underlying stream exhausted
                    break
                if len(probe.buf) >= self.seq_len + 1:
                    produced = True
                    break
            if not produced:
                raise RuntimeError(
                    f"Dataset '{cfg['name']}' (config={cfg.get('config')}) "
                    f"produced no usable sequence in its first "
                    f"~{probe_batches * stream.doc_batch} documents. Check its "
                    "text field / config / data_dir — aborting so its mixture "
                    "weight isn't silently redistributed to the other datasets."
                )
        logger.info("Dataset smoke test passed: every mixture stream yields data.")

    def _select(self):
        r = self._rng.random()
        for i, cw in enumerate(self.cum_weights):
            if r <= cw:
                return i
        return len(self.cum_weights) - 1

    def _next_seq(self):
        if not self.is_mixture:
            # Train indefinitely: on exhaustion, restart with an advanced
            # epoch — a NEW shuffle seed — so the next pass is a different
            # document order, not a replay of the one just finished.
            try:
                return self.streams[0].next_seq()
            except StopIteration:
                self.streams[0].restart()
                logger.info(
                    "Dataset epoch ended; restarting stream at epoch "
                    f"{self.streams[0].epoch} with a fresh shuffle order."
                )
                try:
                    return self.streams[0].next_seq()
                except StopIteration:
                    raise RuntimeError(
                        "Dataset produced no sequences even after a restart; "
                        "check its text field / config."
                    )
        # Cap restart attempts so a dataset that yields no usable sequences
        # (e.g. wrong text field) can't spin the trainer in a tight infinite
        # restart loop with zero progress — skip it and warn instead.
        attempts = 0
        max_attempts = max(2 * len(self.streams), 2)
        while True:
            idx = self._select()
            try:
                return self.streams[idx].next_seq()
            except StopIteration:
                self.streams[idx].restart()
                try:
                    return self.streams[idx].next_seq()
                except StopIteration:
                    attempts += 1
                    logger.warning(
                        f"Dataset '{self.datasets_config[idx]['name']}' yielded "
                        "no usable sequences after restart (check its text "
                        "field); skipping this draw."
                    )
                    if attempts >= max_attempts:
                        raise RuntimeError(
                            "No configured dataset produced any sequences; "
                            "aborting to avoid an infinite restart loop."
                        )
                    continue

    def _produce_batch(self):
        """Pull one batch and snapshot the post-pull data position, atomically.

        Returns ``(batch_dict, data_state_snapshot)`` or ``None`` at epoch end.
        The snapshot is taken under the same lock that advances the streams, so it
        describes the loader EXACTLY after this batch was produced — which is what
        lets a consumer save a data position matching the batch it just trained on
        even though a prefetch worker may have pulled further ahead (see C1).
        """
        with self._lock:
            try:
                seqs = [self._next_seq() for _ in range(self.batch_size)]
            except StopIteration:
                # Defensive only: _next_seq restarts exhausted streams in both
                # single-dataset and mixture mode, so this shouldn't fire.
                return None
            self.batches_consumed += 1
            snapshot = self._snapshot_locked()
        arr = np.array(seqs, dtype=np.int32)
        batch = {
            'input_ids': jnp.array(arr[:, :-1]),
            'labels': jnp.array(arr[:, 1:]),
        }
        return batch, snapshot

    def __iter__(self):
        """Yield batch dicts (no per-batch state). Used by tools/tests that just
        want data; training uses ``iter_with_state`` to track the exact position."""
        while True:
            produced = self._produce_batch()
            if produced is None:
                return
            yield produced[0]

    def iter_with_state(self):
        """Yield ``(batch_dict, data_state_snapshot)`` pairs for resumable training.

        The snapshot rides alongside the batch through the prefetch queue, so when
        training checkpoints after step N it persists the data position as of the
        batch it actually consumed — not the (further-ahead) live loader position.
        """
        while True:
            produced = self._produce_batch()
            if produced is None:
                return
            yield produced

    # ---- checkpoint / resume of the data position ----

    def fast_forward(self, skip_batches):
        """Replay-and-discard ``skip_batches`` batches (fallback resume path).

        Used only when a checkpoint carries no saved data-state. Now backed by
        multithreaded batch tokenization (``_PackedStream`` + ``_encode_batch``),
        so it is materially faster than the old per-document replay, but it still
        re-streams + re-tokenizes the prefix — prefer the saved-state path.
        """
        if skip_batches <= 0:
            return
        logger.info(
            f"No saved data-state; replaying {skip_batches} batches "
            f"({skip_batches * self.batch_size} sequences) to resume position..."
        )
        replayed_seqs = 0
        for i in range(skip_batches * self.batch_size):
            try:
                self._next_seq()
            except StopIteration:
                logger.warning(
                    "Data stream exhausted during fast-forward at "
                    f"{i // self.batch_size}/{skip_batches} batches; "
                    "resuming from stream start."
                )
                break
            replayed_seqs += 1
            if (i + 1) % (self.batch_size * 1000) == 0:
                logger.info(f"  ...replayed {(i + 1) // self.batch_size}/{skip_batches} batches")
        # Record the position actually reached — on early exhaustion, claiming
        # skip_batches would persist a position the stream never hit.
        self.batches_consumed = replayed_seqs // self.batch_size
        logger.info(
            f"Data-stream fast-forward complete ({self.batches_consumed}/"
            f"{skip_batches} batches)."
        )

    def _snapshot_locked(self):
        """Build the data-position snapshot. Caller MUST hold ``self._lock``.

        Kept cheap because it runs on the prefetch worker for EVERY batch even
        though it's only serialized at ``save_interval``: ``rng`` is stored as the
        raw ``getstate()`` tuple (json serializes it as a nested array; rebuilt on
        load) and each stream's ``buf`` is stored by reference, not deep-copied.
        The reference is safe because ``_PackedStream`` only ever rebinds
        ``self.buf`` (never mutates it in place), so a captured list stays frozen
        even as the stream advances.
        """
        return {
            "version": 1,
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "batches_consumed": self.batches_consumed,
            "fingerprint": self._fingerprint,
            "rng": self._rng.getstate(),
            "streams": [
                {"docs_consumed": s.docs_consumed, "buf": s.buf, "epoch": s.epoch}
                for s in self.streams
            ],
        }

    def data_state(self):
        """Snapshot the exact (live) stream position (small, JSON-serializable).

        Note: with a prefetch worker running, this reflects however many batches
        the worker has pulled, which may be ahead of what the consumer has trained
        on. Training therefore persists the per-batch snapshot from
        ``iter_with_state`` instead; this method is for the no-prefetch / tooling
        case (and the older fallback path)."""
        with self._lock:
            return self._snapshot_locked()

    def save_data_state(self, path):
        """Write the live data position next to a checkpoint (best-effort)."""
        _write_data_state(path, self.data_state())

    def load_data_state(self, resume_path):
        """Restore the data position from a checkpoint. Returns True on success.

        On any mismatch/error (older checkpoint without the file, changed
        batch_size/seq_len/mixture, corrupt state) returns False so the caller
        can fall back to ``fast_forward``.
        """
        p = Path(resume_path) / "dataloader_state.json"
        if not p.exists():
            return False
        try:
            with open(p) as f:
                state = json.load(f)
            if (state.get("batch_size") != self.batch_size
                    or state.get("seq_len") != self.seq_len
                    or len(state.get("streams", [])) != len(self.streams)):
                logger.warning(
                    "Saved dataloader state is incompatible with the current "
                    "config (batch_size/seq_len/mixture changed); replaying instead."
                )
                return False
            # Non-fatal determinism check (N6): a changed datasets version or
            # mixture means make_ds().skip() may land on different documents, so
            # the restored position could be misaligned. Warn loudly but still use
            # the saved state — it's the best position available (a replay would
            # depend on the same shuffled order).
            saved_fp = state.get("fingerprint")
            if saved_fp is not None and "stream_skips" not in saved_fp:
                # Checkpoints from before per-stream skips behaved as if every
                # stream skipped its full holdout prefix; normalize so they
                # don't warn spuriously when nothing actually changed.
                saved_fp = dict(saved_fp)
                saved_fp["stream_skips"] = (
                    [saved_fp.get("holdout_examples", 0)] * len(self.streams))
            if saved_fp is not None and saved_fp != self._fingerprint:
                logger.warning(
                    "Dataloader fingerprint changed since this checkpoint "
                    f"(saved={saved_fp}, current={self._fingerprint}). The stream "
                    ".skip() may land on different documents — the resumed data "
                    "position could be misaligned. Continuing with the saved state."
                )
            with self._lock:
                rng = state["rng"]
                self._rng.setstate((rng[0], tuple(rng[1]), rng[2]))
                self.batches_consumed = state["batches_consumed"]
                for stream, ss in zip(self.streams, state["streams"]):
                    stream.resume(
                        ss["docs_consumed"], ss["buf"], ss.get("epoch", 0)
                    )
            logger.info(
                "Resumed data stream from saved state at "
                f"{self.batches_consumed} batches (no re-tokenization)."
            )
            return True
        except Exception as e:
            logger.warning(
                f"Could not load saved dataloader state ({e}); replaying instead."
            )
            return False


def create_jax_dataloader(
    datasets_config,
    dataset_name="wikitext",
    dataset_config="wikitext-103-raw-v1",
    batch_size=8,
    seq_len=512,
    shuffle_buffer=10000,
    seed=42,
    doc_batch=128,
    holdout_examples=0,
):
    """Construct a :class:`StreamingDataLoader` (resume is handled by the caller).

    Returns the loader *object* (iterable) rather than a bare generator so the
    caller can both iterate it and checkpoint/restore its position via
    ``data_state``/``load_data_state``. ``holdout_examples`` reserves the first N
    raw examples of each stream for eval (training skips them; see N1).
    """
    return StreamingDataLoader(
        datasets_config=datasets_config,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        batch_size=batch_size,
        seq_len=seq_len,
        shuffle_buffer=shuffle_buffer,
        seed=seed,
        doc_batch=doc_batch,
        holdout_examples=holdout_examples,
    )


def _load_validation_split(name, config, holdout_examples=10000):
    """Load a dataset's validation split, falling back to the reserved-prefix slice.

    The decision mirrors the training loader exactly via the cached
    ``_has_validation_split`` probe: when a real ``validation`` split exists,
    eval reads it and training reserves NO holdout prefix (no training data
    wasted); when none exists, the training loader skips the first
    ``holdout_examples`` raw examples (``skip_first``) and eval takes exactly
    those (``.take``) — disjoint by construction (N1), regardless of shuffle
    seed or how long training runs. Read unshuffled (a fixed, stable eval
    slice). Returns the dataset iterable, or None on failure.
    """
    if _has_validation_split(name, config):
        try:
            return _load_hf_dataset(name, config, "validation", shuffle_buffer=0)
        except Exception:
            # The probe said a validation split exists, so training reserved NO
            # holdout prefix — falling back to the train prefix here would eval
            # on data training actually sees. Skip this dataset instead.
            return None
    try:
        ds = _load_hf_dataset(name, config, "train", shuffle_buffer=0)
        if holdout_examples > 0:
            ds = ds.take(holdout_examples)
        return ds
    except Exception:
        return None


def create_jax_validation_set(
    datasets_config,
    dataset_name="wikitext",
    dataset_config="wikitext-103-raw-v1",
    num_samples=200,
    seq_len=512,
    batch_size=8,
    holdout_examples=10000,
):
    """
    Create a fixed list of validation batches (JAX arrays).

    When ``datasets_config`` describes a weighted mixture, the validation set is
    drawn from the SAME mixture used in training: each dataset contributes a
    share of ``num_samples`` proportional to its training weight, so eval
    perplexity reflects the actual training distribution rather than just the
    first dataset.

    The held-out data is the reserved prefix (N1): the training loader skips the
    first ``holdout_examples`` raw examples of each stream, and this eval set reads
    only those (``.take(holdout_examples)`` inside ``_load_validation_split``). The
    two are therefore disjoint by construction — no shuffle-window leakage and no
    contamination as training advances — so this eval set is fixed for the whole
    run and ``holdout_examples`` MUST match the value passed to the training loader.

    Returns a list of batch dicts, or None if loading fails.
    """
    tokenizer = _get_tokenizer()

    if datasets_config:
        # Allocate per-dataset sample counts proportional to training weights.
        total_w = sum(d.get("weight", 1.0) for d in datasets_config)
        sequences = []
        for cfg in datasets_config:
            frac = cfg.get("weight", 1.0) / total_w
            quota = max(1, int(round(frac * num_samples)))

            ds = _load_validation_split(
                cfg["name"], cfg.get("config"), holdout_examples=holdout_examples
            )
            if ds is None:
                continue

            count = 0
            for seq in _packed_sequence_iter(iter(ds), tokenizer, seq_len):
                sequences.append(seq)
                count += 1
                if count >= quota:
                    break
    else:
        ds = _load_validation_split(
            dataset_name, dataset_config, holdout_examples=holdout_examples
        )
        if ds is None:
            return None
        sequences = []
        for seq in _packed_sequence_iter(iter(ds), tokenizer, seq_len):
            sequences.append(seq)
            if len(sequences) >= num_samples:
                break

    if not sequences:
        return None

    batches = []
    n_full = len(sequences) // batch_size
    if n_full == 0:
        # Fewer than one full batch: emit a single (smaller) batch. There's only
        # one shape here, so still just one eval_step compile.
        chunks = [sequences]
    else:
        # Keep only full batches; drop the ragged remainder. A short final batch
        # is a second tensor shape that forces an extra eval_step XLA compile
        # (and recompiles on every eval) for negligible eval coverage (M3).
        chunks = [
            sequences[i:i + batch_size]
            for i in range(0, n_full * batch_size, batch_size)
        ]
    for chunk in chunks:
        arr = np.array(chunk, dtype=np.int32)
        batches.append({
            'input_ids': jnp.array(arr[:, :-1]),
            'labels': jnp.array(arr[:, 1:]),
        })
    return batches


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def main():
    global logger
    logger = setup_logging()
    
    parser = argparse.ArgumentParser(description="Train 500M model on v5e TPU")
    parser.add_argument("-c", "--config", type=str, required=True, help="Config YAML")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--auto-resume", action="store_true", help="Auto-resume from latest")
    args = parser.parse_args()
    
    logger.info(
        "train_v5e_complete.py — v3 (audit fixes: graceful signal stop, "
        "endless single-dataset epochs, block remat, holdout only when needed)"
    )
    
    # Setup Google Drive
    setup_google_drive()
    
    # Load configs
    logger.info(f"Loading config from {args.config}")
    model_config, training_config = load_config(args.config)
    
    logger.info("\n=== Model Config ===")
    for k, v in model_config.items():
        logger.info(f"  {k}: {v}")
    
    logger.info("\n=== Training Config ===")
    for k, v in training_config.items():
        if k not in ['datasets']:
            logger.info(f"  {k}: {v}")
    
    # Create output directory
    output_dir = training_config['output_dir']
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Device visibility: this script is single-device (jit, no pmap/sharding) —
    # make that an explicit, logged choice rather than a silent one.
    logger.info(
        f"\nJAX backend: {jax.default_backend()} | devices visible: "
        f"{jax.device_count()}"
    )
    if jax.device_count() > 1:
        logger.warning(
            f"{jax.device_count()} devices visible but this script trains on a "
            f"single device ({jax.devices()[0]}); the rest will sit idle."
        )

    # Initialize model
    logger.info("\nInitializing model...")
    key = jax.random.PRNGKey(42)
    model = TransformerModel(model_config, key)
    # Rows per chunk for the memory-efficient cross-entropy (compute_loss). Folds
    # the vocab projection into a rematerialized, chunked CE so the full
    # (batch*seq, vocab) f32 logits are never held in HBM — the main lever on how
    # large batch_size can grow on a 16 GB chip.
    # NOTE: compute_loss unrolls its chunk loop into the traced graph (and,
    # with gradient accumulation, into the scan body) — a tiny chunk size means
    # many unrolled (rows, vocab) matmuls and a long XLA compile. A warning
    # fires below once seq_len is known.
    model.ce_chunk_size = int(training_config.get('ce_chunk_size', 1024))
    # Block remat: recompute each block's forward during backward instead of
    # holding ~n_layers blocks of activations (several GB at 500M scale — the
    # real batch-size cap once the CE is chunked). Same memory/recompute lever
    # as the chunked CE. Disable with `remat_blocks: false`.
    model.remat_blocks = bool(training_config.get('remat_blocks', True))
    total_params = model.count_parameters()
    logger.info(f"Total parameters: {total_params:,} ({total_params / 1e6:.1f}M)")

    # Keep an f32 master copy of the weights. The forward pass casts to bf16 on
    # the fly (see `_loss_fn`), but AdamW accumulates updates against this f32
    # master so that late-training updates smaller than a bf16 ULP aren't lost
    # to rounding (which would stall learning). Costs ~+1.6 GB; m/v are already
    # f32, so this is standard mixed-precision practice.
    model.params = jax.tree_util.tree_map(
        lambda p: p.astype(jnp.float32), model.params
    )
    
    # Initialize optimizer (moment buffers are allocated AFTER the resume
    # decision below — see there).
    optimizer = AdamWOptimizer(
        beta1=training_config['betas'][0],
        beta2=training_config['betas'][1],
        eps=training_config['eps'],
        weight_decay=training_config['weight_decay'],
    )
    
    # Resume from checkpoint
    start_step = 0
    old_lr = None
    data_pos = 0
    resume_path = None
    if args.resume:
        logger.info(f"\nResuming from {args.resume}")
        resume_path = args.resume
        start_step, old_lr, data_pos = load_checkpoint(
            args.resume, model, optimizer, model_config, training_config
        )
    elif args.auto_resume:
        latest = get_latest_checkpoint(output_dir)
        if latest:
            logger.info(f"\nAuto-resuming from {latest}")
            resume_path = latest
            start_step, old_lr, data_pos = load_checkpoint(
                latest, model, optimizer, model_config, training_config
            )
    # Allocate zero moment buffers only when a resume didn't load them — doing
    # it before the resume would transiently hold both the zeros AND the loaded
    # state on device (~4 GB at 500M scale) for nothing.
    if not optimizer.m:
        optimizer.init_state(model.params)

    # ----------------------------------------------------------------
    # Smooth LR ramp on resume.
    # If the schedule changed since the checkpoint was written (e.g. max_steps
    # was enlarged), the new cosine curve would jump at the resume point. Ramp
    # linearly from the LR the checkpoint was trained at to where the NEW curve
    # will be `ramp_steps` later, then follow the new curve. Skipped when the
    # discrepancy is small (a same-schedule resume) or the old LR is unknown.
    # ----------------------------------------------------------------
    if old_lr is not None and start_step > 0:
        ramp_steps = int(training_config.get('lr_resume_ramp_steps', 2000))
        threshold = float(training_config.get('lr_resume_ramp_threshold', 0.1))
        new_curve_now = get_learning_rate(start_step, training_config)
        rel_diff = abs(new_curve_now - old_lr) / max(old_lr, 1e-12)
        if ramp_steps > 0 and rel_diff > threshold:
            # Target = where the NEW curve will be at the end of the ramp.
            target_lr = get_learning_rate(start_step + ramp_steps, training_config)
            training_config['lr_ramp_start_step'] = start_step
            training_config['lr_ramp_steps'] = ramp_steps
            training_config['lr_ramp_from'] = old_lr
            training_config['lr_ramp_to'] = target_lr
            logger.info(
                f"LR resume-ramp ENABLED: {old_lr:.3e} -> {target_lr:.3e} over "
                f"{ramp_steps} steps (new-curve LR at resume would be "
                f"{new_curve_now:.3e}, {rel_diff * 100:.1f}% off old LR)"
            )
        else:
            logger.info(
                f"LR resume-ramp skipped (old={old_lr:.3e}, "
                f"new-curve={new_curve_now:.3e}, {rel_diff * 100:.1f}% diff "
                f"<= {threshold * 100:.0f}% threshold)"
            )

    logger.info(f"\nStarting training from step {start_step}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("-" * 60)
    
    # Training hyperparameters
    step = start_step
    max_steps = training_config.get('max_steps', 1000000)
    save_interval = training_config.get('save_interval', 5000)
    eval_interval = training_config.get('eval_interval', 1000)
    log_interval = training_config.get('log_interval', 20)
    # If loss/grad stays non-finite for this many CONSECUTIVE steps the run has
    # diverged; save one last (last-good-weights) checkpoint and hard-abort so a
    # dead run doesn't keep burning TPU credits. 0 disables the auto-abort.
    nonfinite_abort_steps = int(training_config.get('nonfinite_abort_steps', 50))
    consecutive_nonfinite = 0

    # Loss/grad-norm are kept on-device and only pulled to the host once per
    # log_interval (see P1 below), so we buffer the per-step device scalars here
    # rather than syncing each step.
    pending_losses = []
    pending_grad_norms = []
    tokens_processed = 0
    start_time = time.time()
    last_log_time = start_time
    
    # ----------------------------------------------------------------
    # Create streaming data loader (JAX-compatible)
    # ----------------------------------------------------------------
    logger.info("\nCreating data loaders...")
    # Reserved-prefix eval holdout (N1): training skips the first N raw examples of
    # each stream; the eval set reads exactly those. Both sides MUST use the same N.
    holdout_examples = int(training_config.get('val_holdout_examples', 10000))
    # Sequence length for data packing and the model's effective context.
    # `training.sequence_length` lets you train on shorter sequences than the
    # architecture's `model.max_seq_len` — RoPE tables and the causal mask are
    # built per-call from the ACTUAL input length, so seq_len <= max_seq_len is
    # valid and needs no architecture change. Defaults to max_seq_len when unset;
    # a value above max_seq_len is clamped (and warned) rather than silently used.
    max_seq_len = model_config.get('max_seq_len', 512)
    seq_len = int(training_config.get('sequence_length', max_seq_len))
    if seq_len > max_seq_len:
        logger.warning(
            f"training.sequence_length ({seq_len}) exceeds model.max_seq_len "
            f"({max_seq_len}); clamping to max_seq_len."
        )
        seq_len = max_seq_len
    # Tiny CE chunks unroll into a huge traced graph (see ce_chunk_size note
    # above): warn early rather than mid-compile.
    _ce_rows = int(training_config.get('batch_size', 8)) * seq_len
    if model.ce_chunk_size > 0:
        _n_chunks = -(-_ce_rows // model.ce_chunk_size)  # ceil div
        if _n_chunks > 32:
            logger.warning(
                f"ce_chunk_size={model.ce_chunk_size} unrolls into {_n_chunks} "
                "CE chunks per micro-batch — expect a long XLA compile."
            )
    data_loader = create_jax_dataloader(
        datasets_config=training_config.get('datasets'),
        dataset_name=training_config.get('dataset_name', 'wikitext'),
        dataset_config=training_config.get('dataset_config', 'wikitext-103-raw-v1'),
        batch_size=training_config.get('batch_size', 8),
        seq_len=seq_len,
        shuffle_buffer=int(training_config.get('shuffle_buffer', 10000)),
        seed=int(training_config.get('seed', 42)),
        doc_batch=int(training_config.get('tokenize_doc_batch', 128)),
        holdout_examples=holdout_examples,
    )
    # Out-of-range token ids are silently CLAMPED by JAX's gather — a
    # vocab_size smaller than the tokenizer's would alias tokens onto the last
    # embedding row and corrupt training invisibly. Fail loudly instead.
    if model.vocab_size < data_loader.tokenizer.n_vocab:
        raise ValueError(
            f"model.vocab_size ({model.vocab_size}) < tokenizer vocabulary "
            f"({data_loader.tokenizer.n_vocab}); out-of-range token ids would "
            "be silently clamped. Increase model.vocab_size."
        )
    # Fail loudly at startup if any mixture dataset yields no usable sequences
    # (wrong text field / missing config / data_dir) instead of silently
    # redistributing its weight to the others at run time (N2). Probes throwaway
    # streams, so it's safe to run before restoring the resume position below.
    if training_config.get('validate_datasets', True):
        data_loader.validate_streams()

    # Restore the exact data position on resume. Prefer the position saved in the
    # checkpoint (no re-tokenization); fall back to replaying the stream only for
    # older checkpoints that predate data-state saving.
    if resume_path is not None and data_loader.load_data_state(resume_path):
        pass
    elif data_pos > 0:
        data_loader.fast_forward(data_pos)
    # Overlap tokenization + host→device transfer with on-device compute so the
    # TPU isn't stalled waiting on the single-threaded Python data pipeline.
    # iter_with_state pairs each batch with the data position as of that batch, so
    # the prefetch worker running ahead doesn't desync the saved data position
    # from the trained step (C1).
    train_loader = prefetch(data_loader.iter_with_state(), depth=3)

    val_loader = create_jax_validation_set(
        datasets_config=training_config.get('datasets'),
        dataset_name=training_config.get('dataset_name', 'wikitext'),
        dataset_config=training_config.get('dataset_config', 'wikitext-103-raw-v1'),
        num_samples=int(training_config.get('val_num_samples', 200)),
        seq_len=seq_len,
        batch_size=training_config.get('batch_size', 8),
        # Read exactly the reserved prefix that the training loader skips, so eval
        # is disjoint from training by construction (N1). Same N on both sides.
        holdout_examples=holdout_examples,
    )
    if not val_loader:
        logger.warning(
            "Validation set unavailable (no split yielded any sequences); "
            "eval is DISABLED for this run."
        )

    gradient_clip = training_config.get('gradient_clip', 1.0)
    # Gradient accumulation (N4): run `accum_steps` micro-batches per optimizer
    # step and SUM their grads inside train_step via lax.scan. Because scan is
    # sequential, only one micro-batch's activations are ever live, so the
    # effective batch grows with NO extra activation memory — the memory-free
    # lever the audits called out. 1 = disabled (still a length-1 scan).
    accum_steps = max(1, int(training_config.get('gradient_accumulation_steps', 1)))
    _bs = training_config.get('batch_size', 8)
    _sl = seq_len
    if accum_steps > 1:
        logger.info(
            f"Gradient accumulation ON: {accum_steps} micro-batches/step "
            f"-> effective batch {accum_steps * _bs} seqs "
            f"({accum_steps * _bs * _sl} tokens/step)."
        )

    # Build closures that capture `model` (a static Python object) so that
    # JAX only traces the pure-function parts (params & batch).
    def _loss_fn(params, batch):
        # `params` is the f32 master copy. Cast to bf16 for the forward pass so
        # attention/matmuls stay fast and memory-cheap; the cast is differentiable
        # (straight-through), so gradients — and thus the AdamW update — still
        # accumulate against the full-precision master.
        params_bf16 = jax.tree_util.tree_map(
            lambda p: p.astype(jnp.bfloat16), params
        )
        return compute_loss(params_bf16, batch, model)

    # Optimizer hyperparameters are captured as static constants; everything
    # that varies per step (params, moments, step count, lr, batch) is traced.
    opt_beta1, opt_beta2 = optimizer.beta1, optimizer.beta2
    opt_eps, opt_wd = optimizer.eps, optimizer.weight_decay

    # Exclude 1-D norm gains (and, when configured, embeddings) from weight
    # decay. The mask is a static pytree of 0.0/1.0 captured by `train_step`, so
    # XLA bakes it in as constants — no runtime cost. `weight_decay_embeddings`
    # defaults True (decay embeddings, GPT-style); norms are always excluded.
    decay_mask = build_decay_mask(
        model.params,
        decay_embeddings=training_config.get('weight_decay_embeddings', True),
    )

    # Donate the params/m/v buffers (argnums 0,1,2): they are replaced wholesale
    # by the step's outputs, so XLA can reuse their HBM in-place instead of
    # allocating fresh buffers and copying — saves ~1.6 GB and avoids the extra
    # copies. (t, lr, batch are not donated: t/lr are tiny and batch is reused
    # only as input.)
    @partial(jit, donate_argnums=(0, 1, 2))
    def train_step(params, m, v, t, lr, batch):
        """Fused fwd + bwd + grad-accum + grad-clip + AdamW update — one XLA program.

        Folding the optimizer into the same jit as the loss/grad computation
        means the entire step is a single kernel launch: no per-parameter Python
        dispatch (the previous optimizer loop cost ~150 dispatches/step) and no
        round-trip of the gradient pytree between two separate compiled programs.

        ``batch`` carries a leading axis of ``accum_steps`` micro-batches (N4).
        Their grads are SUMMED with ``lax.scan`` — sequential, so only ONE
        micro-batch's activations are live at a time and the effective batch grows
        with no extra activation memory — then averaged and applied once.
        ``accum_steps == 1`` is just a length-1 scan (one extra zero-add of the
        grad tree). The reported ``loss`` is the mean over the micro-batches.
        """
        def micro(grad_acc, mb):
            loss_i, grads_i = value_and_grad(_loss_fn)(params, mb)
            grad_acc = jax.tree_util.tree_map(lambda a, g: a + g, grad_acc, grads_i)
            return grad_acc, loss_i

        zero_grads = jax.tree_util.tree_map(jnp.zeros_like, params)
        grad_acc, losses = lax.scan(micro, zero_grads, batch)
        n_micro = losses.shape[0]
        grads = jax.tree_util.tree_map(lambda g: g / n_micro, grad_acc)
        loss = jnp.mean(losses)

        grads, grad_norm = clip_gradients(grads, gradient_clip)
        # Tentative AdamW step count for THIS update (committed below only if the
        # step is finite, so the bias correction never advances on a skipped one).
        t_new = t + 1
        upd_params, upd_m, upd_v = AdamWOptimizer.apply(
            params, grads, m, v, t_new, lr,
            opt_beta1, opt_beta2, opt_eps, opt_wd, decay_mask,
        )
        # Non-finite guard (C2): a single NaN/Inf loss or gradient would write
        # NaN into the f32 master and poison the whole run (and any checkpoint
        # saved afterwards). If this step isn't finite, keep params/m/v AND the
        # step count `t` unchanged (N3) so a transient spike self-heals, the last
        # good state is what later gets checkpointed, and the bias correction
        # doesn't drift on a no-op update. The host loop counts consecutive
        # non-finite steps (via the synced loss/grad_norm) and hard-aborts if it
        # persists. The selects add a little elementwise traffic every step, but
        # it's tiny next to the fwd/bwd matmuls and buys robustness.
        finite = jnp.isfinite(loss) & jnp.isfinite(grad_norm)
        keep_if_bad = lambda new, old: jax.tree_util.tree_map(
            lambda a, b: jnp.where(finite, a, b), new, old
        )
        new_params = keep_if_bad(upd_params, params)
        new_m = keep_if_bad(upd_m, m)
        new_v = keep_if_bad(upd_v, v)
        new_t = jnp.where(finite, t_new, t)
        return new_params, new_m, new_v, new_t, loss, grad_norm

    @jit
    def eval_step(params, batch):
        """JIT-compiled evaluation step."""
        return _loss_fn(params, batch)

    # ----------------------------------------------------------------
    # Graceful stop. The first SIGINT/SIGTERM only sets a flag: the loop
    # finishes the in-flight step, checkpoints, and exits cleanly. Letting the
    # default KeyboardInterrupt fire mid-step is NOT safe here — train_step
    # donates params/m/v, so an exception between dispatch and reassignment can
    # strand model.params on deleted buffers and make even a rescue-save crash.
    # A second signal restores the default handler for an immediate exit.
    # ----------------------------------------------------------------
    _stop_signal = {"num": None}

    def _request_stop(signum, frame):
        if _stop_signal["num"] is not None:
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
        _stop_signal["num"] = signum
        # print, not logger: the logging module isn't reentrancy-safe inside
        # signal handlers.
        print(
            f"\nSignal {signum} received — will checkpoint and exit after the "
            "current step (send again to force-quit)."
        )

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    # ----------------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------------
    logger.info("\nStarting training loop...")
    logger.info(f"  batch_size={training_config.get('batch_size', 8)}, "
                f"seq_len={seq_len}, "
                f"grad_clip={gradient_clip}")
    logger.info("-" * 60)

    # Holds the data-position snapshot for the batch most recently TRAINED on (not
    # merely pulled from the prefetch queue), so a checkpoint persists the position
    # matching the trained step (see C1). It's updated only AFTER train_step, so
    # the batch pulled-but-discarded when the loop breaks on max_steps doesn't
    # advance the saved position one batch past the trained step (M1). None until
    # the first batch (covers the degenerate "already at max_steps" case).
    data_snapshot = None
    interrupted = False
    train_iter = iter(train_loader)

    # Background checkpoint writer: the periodic in-loop saves copy state to host
    # on this thread, then hand the slow compress+Drive-write to the worker so the
    # TPU keeps training instead of stalling minutes per save (the abort / signal /
    # final saves stay synchronous via save_checkpoint + saver.flush()).
    saver = CheckpointSaver()

    while step < max_steps:
        # Graceful stop: the checkpoint-and-exit is handled after the loop.
        if _stop_signal["num"] is not None:
            interrupted = True
            break
        # Pull `accum_steps` micro-batches and stack them along a new leading axis
        # for train_step's scan (N4). The snapshot of the LAST micro-batch is the
        # furthest-consumed data position — what a resume must continue from once
        # the accumulated update is applied.
        micro_inputs, micro_labels = [], []
        pulled_snapshot = None
        try:
            for _ in range(accum_steps):
                mb, mb_snap = next(train_iter)
                micro_inputs.append(mb['input_ids'])
                micro_labels.append(mb['labels'])
                pulled_snapshot = mb_snap
        except StopIteration:
            # Defensive: streams now restart forever, so the prefetch generator
            # only ends if the producer errored (which re-raises, not this).
            break
        batch = {
            'input_ids': jnp.stack(micro_inputs),
            'labels': jnp.stack(micro_labels),
        }

        # Learning rate schedule. step+1 = the index of the step this update
        # produces: without it the first warmup step trains at LR=0 (wasting a
        # batch while still advancing Adam's moments) and every checkpoint's
        # recorded LR (computed at the post-increment step) is one step ahead
        # of the LR actually used.
        lr = get_learning_rate(step + 1, training_config)

        # Fully fused step: fwd + bwd + grad-accum + clip + AdamW, all inside JIT.
        # The optimizer's moment buffers + step count live on the instance so
        # checkpointing still sees them. `optimizer.t` is advanced INSIDE
        # train_step and only on finite steps (N3); it comes back as a device
        # scalar (synced to int only at checkpoint time, never per step).
        model.params, optimizer.m, optimizer.v, optimizer.t, loss, grad_norm = train_step(
            model.params, optimizer.m, optimizer.v, optimizer.t, lr, batch
        )
        # These micro-batches have now been trained on; record the furthest data
        # position so any checkpoint below (or the final save) matches it (M1).
        data_snapshot = pulled_snapshot
        # P1: do NOT force a device→host sync here. `float(loss)` every step
        # serializes the pipeline — JAX can't dispatch step N+1 while N is still
        # running. Instead keep the scalars on-device and transfer the whole
        # interval's worth in a single round-trip at the log boundary below.
        pending_losses.append(loss)
        pending_grad_norms.append(grad_norm)
        # batch['input_ids'] is (accum_steps, batch_size, seq_len).
        _ii = batch['input_ids']
        tokens_processed += _ii.shape[0] * _ii.shape[1] * _ii.shape[2]
        step += 1

        if step == start_step + 1:
            # The first step pays the one-off XLA compile of train_step. Reset the
            # throughput window so the first logged tok/s isn't dominated by it.
            tokens_processed = 0
            last_log_time = time.time()

        # ----- Logging -----
        if step % log_interval == 0:
            # One host sync for the entire interval (P1).
            losses = np.asarray(jax.device_get(jnp.stack(pending_losses)))
            grad_norms = np.asarray(jax.device_get(jnp.stack(pending_grad_norms)))

            # Non-finite watch (C2). train_step already SKIPS the weight update on
            # any non-finite step, so params are never poisoned; here we just count
            # how long the divergence persists. Walk the interval's steps in order
            # so the consecutive run carries across log boundaries.
            finite_mask = np.isfinite(losses) & np.isfinite(grad_norms)
            for ok in finite_mask:
                consecutive_nonfinite = 0 if ok else consecutive_nonfinite + 1
            n_bad = int((~finite_mask).sum())

            # Average/perplexity over the FINITE steps only (a single NaN would
            # otherwise turn the whole interval's mean into NaN).
            finite_losses = losses[finite_mask]
            avg_loss = float(finite_losses.mean()) if finite_losses.size else float('nan')
            # Max (not last) over the interval — surfaces grad spikes that a
            # single sampled step would miss.
            finite_norms = grad_norms[finite_mask]
            grad_norm_val = float(finite_norms.max()) if finite_norms.size else float('nan')
            ppl = compute_perplexity(avg_loss)
            now = time.time()
            elapsed = now - last_log_time
            tok_per_sec = tokens_processed / elapsed if elapsed > 0 else 0

            logger.info(
                f"Step {step:6d} | Loss: {avg_loss:.4f} | PPL: {ppl:.2f} | "
                f"LR: {lr:.2e} | Grad: {grad_norm_val:.3f} | "
                f"Tok/s: {tok_per_sec:.0f}"
            )
            if n_bad:
                logger.warning(
                    f"Non-finite loss/grad on {n_bad}/{len(finite_mask)} steps this "
                    f"interval; their updates were skipped "
                    f"(consecutive run: {consecutive_nonfinite})."
                )

            pending_losses = []
            pending_grad_norms = []
            tokens_processed = 0
            last_log_time = now

            # Persistent divergence: save the last good weights and hard-abort so
            # a dead run stops burning credits. SIGKILL can't be caught and does
            # no cleanup, so the checkpoint MUST be written first.
            if nonfinite_abort_steps and consecutive_nonfinite >= nonfinite_abort_steps:
                logger.error(
                    f"Loss/grad non-finite for {consecutive_nonfinite} consecutive "
                    f"steps (>= nonfinite_abort_steps={nonfinite_abort_steps}). "
                    "Training has diverged; saving a final checkpoint with the last "
                    "finite weights and hard-aborting."
                )
                # Let any in-flight async periodic save finish before we take the
                # uncatchable SIGKILL path (a killed background thread would leave
                # a .tmp dir, never a published checkpoint, but flushing keeps the
                # last good periodic checkpoint intact).
                saver.flush()
                abort_path = str(Path(output_dir) / f"step_{step}_nonfinite_abort")
                save_checkpoint(
                    path=abort_path,
                    params=model.params,
                    optimizer=optimizer,
                    step=step,
                    model_config=model_config,
                    training_config=training_config,
                    data_loader=data_loader,
                    data_state=data_snapshot,
                )
                logging.shutdown()  # flush/close file handlers before SIGKILL
                sys.stdout.flush()
                os.kill(os.getpid(), signal.SIGKILL)  # signal 9: immediate, uncatchable

        # ----- Evaluation -----
        if step % eval_interval == 0 and val_loader:
            logger.info("\nRunning evaluation...")
            # Collect per-batch losses on device, then sync once. A per-batch
            # float(vl) would force ~25 device->host syncs that serialize eval.
            val_losses = []
            for val_batch in val_loader:
                val_losses.append(eval_step(model.params, val_batch))
            val_loss = (
                float(np.mean(jax.device_get(jnp.stack(val_losses))))
                if val_losses else 0.0
            )
            val_ppl = compute_perplexity(val_loss)
            logger.info(f"  Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}\n")

        # ----- Checkpointing (async) -----
        if step % save_interval == 0:
            ckpt_path = str(Path(output_dir) / f"step_{step}")
            # Copy params/opt-state to host on THIS thread (a brief device→host
            # transfer), then hand the slow compress + Drive write (and the prune)
            # to the background saver so the TPU keeps training. The payload is an
            # independent host snapshot, so the next train_step may donate the live
            # buffers freely.
            payload = _checkpoint_payload(
                model.params, optimizer, step, model_config,
                training_config, data_loader, data_snapshot,
            )
            saver.submit(
                ckpt_path, payload,
                prune_args=(output_dir, step, save_interval),
            )

    # Wait for any in-flight async periodic save (and its prune) to finish before
    # the synchronous final/interrupt save below, so the two never race on the
    # output dir and no background write is dropped at process exit.
    saver.flush()

    # Final checkpoint.
    if interrupted:
        # Graceful signal stop: persist progress under a step-named checkpoint
        # (NOT "final" — the run didn't complete) and exit.
        if step > start_step:
            stop_path = Path(output_dir) / f"step_{step}"
            if stop_path.exists():
                logger.info(
                    f"Stop requested; checkpoint for step {step} already "
                    "exists — skipping re-save."
                )
            else:
                logger.info(
                    f"Stop requested (signal {_stop_signal['num']}); saving "
                    f"checkpoint at step {step} before exit."
                )
                save_checkpoint(
                    path=str(stop_path),
                    params=model.params,
                    optimizer=optimizer,
                    step=step,
                    model_config=model_config,
                    training_config=training_config,
                    data_loader=data_loader,
                    data_state=data_snapshot,
                )
        else:
            logger.info(
                "Stop requested before any step completed; nothing new to save."
            )
        logger.info("Exiting on signal.")
    elif step == start_step:
        # Nothing was trained (e.g. resumed a run already at max_steps). Skip
        # the final save: data_snapshot is None here, and the fallback would
        # persist the LIVE loader position — which the prefetch worker has
        # already advanced a few batches past the last trained batch.
        logger.info(
            "\nNo steps were trained this run; skipping the final checkpoint."
        )
    else:
        logger.info("\nTraining complete!")
        final_path = str(Path(output_dir) / "final")
        save_checkpoint(
            path=final_path,
            params=model.params,
            optimizer=optimizer,
            step=step,
            model_config=model_config,
            training_config=training_config,
            data_loader=data_loader,
            data_state=data_snapshot,
        )
    total_time = time.time() - start_time
    logger.info(f"Total training time: {total_time / 3600:.2f} hours")


if __name__ == "__main__":
    main()
