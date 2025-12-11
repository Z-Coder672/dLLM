"""
Multi-Head Attention with Rotary Position Embeddings (RoPE).

Uses ternary linear layers for Q, K, V, and output projections.
"""

import mlx.core as mx
import mlx.nn as nn
import math
from typing import Optional, Tuple

from .layers import TernaryLinear


def precompute_rope_frequencies(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    dtype: mx.Dtype = mx.bfloat16
) -> Tuple[mx.array, mx.array]:
    """
    Precompute RoPE frequency tensors.
    
    Args:
        dim: Dimension of each head
        max_seq_len: Maximum sequence length
        theta: Base for frequency computation
        dtype: Output dtype
        
    Returns:
        cos, sin: Frequency tensors of shape (max_seq_len, dim)
    """
    # Compute frequencies
    freqs = 1.0 / (theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
    
    # Compute position indices
    positions = mx.arange(max_seq_len, dtype=mx.float32)
    
    # Outer product: (seq_len,) x (dim/2,) -> (seq_len, dim/2)
    angles = positions[:, None] * freqs[None, :]
    
    # Compute cos and sin, then interleave to get full dimension
    cos = mx.cos(angles)
    sin = mx.sin(angles)
    
    # Repeat to match full dimension: (seq_len, dim/2) -> (seq_len, dim)
    cos = mx.repeat(cos, 2, axis=-1)
    sin = mx.repeat(sin, 2, axis=-1)
    
    return cos.astype(dtype), sin.astype(dtype)


def apply_rope(
    x: mx.array,
    cos: mx.array,
    sin: mx.array,
    offset: int = 0
) -> mx.array:
    """
    Apply Rotary Position Embeddings to input tensor.
    
    Args:
        x: Input tensor, shape (batch, seq_len, n_heads, head_dim)
        cos: Cosine frequencies, shape (max_seq_len, head_dim)
        sin: Sine frequencies, shape (max_seq_len, head_dim)
        offset: Position offset for incremental decoding
        
    Returns:
        Rotated tensor with same shape as input
    """
    seq_len = x.shape[1]
    
    # Get relevant positions
    cos = cos[offset:offset + seq_len]  # (seq_len, head_dim)
    sin = sin[offset:offset + seq_len]
    
    # Reshape for broadcasting: (1, seq_len, 1, head_dim)
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    
    # Rotate pairs of features
    # Split into pairs and rotate
    x_reshape = x.reshape(*x.shape[:-1], -1, 2)
    x1 = x_reshape[..., 0]
    x2 = x_reshape[..., 1]
    
    # Apply rotation
    cos_reshape = cos.reshape(*cos.shape[:-1], -1, 2)[..., 0]
    sin_reshape = sin.reshape(*sin.shape[:-1], -1, 2)[..., 0]
    
    rotated_x1 = x1 * cos_reshape - x2 * sin_reshape
    rotated_x2 = x1 * sin_reshape + x2 * cos_reshape
    
    # Interleave back
    rotated = mx.stack([rotated_x1, rotated_x2], axis=-1)
    return rotated.reshape(x.shape)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with RoPE and ternary projections.
    
    Uses grouped query attention (GQA) option for efficiency.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        threshold_factor: float = 0.7,
        max_seq_len: int = 2048,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        
        self.n_rep = n_heads // self.n_kv_heads  # Repetition factor for GQA
        
        # Projections using ternary linear layers
        self.q_proj = TernaryLinear(d_model, n_heads * self.head_dim, threshold_factor=threshold_factor)
        self.k_proj = TernaryLinear(d_model, self.n_kv_heads * self.head_dim, threshold_factor=threshold_factor)
        self.v_proj = TernaryLinear(d_model, self.n_kv_heads * self.head_dim, threshold_factor=threshold_factor)
        self.o_proj = TernaryLinear(n_heads * self.head_dim, d_model, threshold_factor=threshold_factor)
        
        # Precompute RoPE frequencies
        self.cos, self.sin = precompute_rope_frequencies(
            self.head_dim, max_seq_len, rope_theta, mx.bfloat16
        )
        
        # Scaling factor for attention
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
        training: bool = False,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        """
        Forward pass for multi-head attention.
        
        Args:
            x: Input tensor, shape (batch, seq_len, d_model)
            mask: Attention mask, shape (batch, 1, seq_len, seq_len)
            cache: Optional KV cache tuple (k_cache, v_cache)
            training: Whether in training mode
            
        Returns:
            output: Attention output, shape (batch, seq_len, d_model)
            new_cache: Updated KV cache if cache was provided
        """
        batch_size, seq_len, _ = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape to (batch, seq_len, n_heads, head_dim)
        q = q.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        
        # Apply RoPE
        offset = 0 if cache is None else cache[0].shape[1]
        q = apply_rope(q, self.cos, self.sin, offset)
        k = apply_rope(k, self.cos, self.sin, offset)
        
        # Handle KV cache
        if cache is not None:
            k_cache, v_cache = cache
            k = mx.concatenate([k_cache, k], axis=1)
            v = mx.concatenate([v_cache, v], axis=1)
        new_cache = (k, v) if cache is not None else None
        
        # Repeat K, V for grouped query attention
        if self.n_rep > 1:
            k = mx.repeat(k, self.n_rep, axis=2)
            v = mx.repeat(v, self.n_rep, axis=2)
        
        # Transpose to (batch, n_heads, seq_len, head_dim)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        
        # Compute attention scores
        # (batch, n_heads, q_len, head_dim) @ (batch, n_heads, head_dim, kv_len)
        # -> (batch, n_heads, q_len, kv_len)
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        
        # Apply mask
        if mask is not None:
            scores = scores + mask
        
        # Softmax and dropout
        attn_weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(x.dtype)
        
        if training and self.dropout > 0:
            attn_weights = mx.dropout(attn_weights, p=self.dropout)
        
        # Apply attention to values
        # (batch, n_heads, q_len, kv_len) @ (batch, n_heads, kv_len, head_dim)
        # -> (batch, n_heads, q_len, head_dim)
        attn_output = attn_weights @ v
        
        # Reshape back to (batch, seq_len, d_model)
        attn_output = attn_output.transpose(0, 2, 1, 3)
        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        
        # Output projection
        output = self.o_proj(attn_output)
        
        return output, new_cache


def create_causal_mask(seq_len: int, dtype: mx.Dtype = mx.bfloat16) -> mx.array:
    """
    Create a causal attention mask.
    
    Args:
        seq_len: Sequence length
        dtype: Output dtype
        
    Returns:
        Mask tensor of shape (1, 1, seq_len, seq_len) with -inf for masked positions
    """
    # Create upper triangular mask (1s above diagonal)
    mask = mx.triu(mx.ones((seq_len, seq_len)), k=1)
    
    # Convert to -inf for masked positions
    mask = mx.where(mask, mx.array(float("-inf")), mx.array(0.0))
    
    # Add batch and head dimensions
    return mask[None, None, :, :].astype(dtype)

