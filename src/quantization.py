"""
INT8 quantization utilities for shadow weights.

Per-channel symmetric quantization:
- Scale computed per output channel (row)
- Values clipped to [-127, 127]
- Zero point is always 0 (symmetric)
"""

import mlx.core as mx
from typing import Tuple


def quantize_int8(w_bf16: mx.array) -> Tuple[mx.array, mx.array]:
    """
    Quantize BF16 weights to INT8 with per-channel scaling.
    
    Args:
        w_bf16: Weight tensor in bfloat16, shape (out_features, in_features)
        
    Returns:
        w_int8: Quantized weights in int8
        scale: Per-channel scale factors in bfloat16, shape (out_features, 1)
    """
    # Compute per-channel (per-row) scale
    # Add small epsilon to avoid division by zero
    abs_max = mx.max(mx.abs(w_bf16), axis=-1, keepdims=True)
    scale = abs_max / 127.0 + 1e-8
    
    # Quantize: round and clip to int8 range
    w_scaled = w_bf16 / scale
    w_int8 = mx.clip(mx.round(w_scaled), -127, 127).astype(mx.int8)
    
    return w_int8, scale.astype(mx.bfloat16)


def dequantize_int8(w_int8: mx.array, scale: mx.array) -> mx.array:
    """
    Dequantize INT8 weights back to BF16.
    
    Args:
        w_int8: Quantized weights in int8
        scale: Per-channel scale factors in bfloat16
        
    Returns:
        w_bf16: Dequantized weights in bfloat16
    """
    return w_int8.astype(mx.bfloat16) * scale


def update_scales(w_int8: mx.array, scale: mx.array, w_bf16_updated: mx.array) -> Tuple[mx.array, mx.array]:
    """
    Update INT8 weights and scales after optimizer step.
    
    This is called after the optimizer updates the dequantized weights.
    We re-quantize back to INT8 with updated scales.
    
    Args:
        w_int8: Current INT8 weights (unused, for reference)
        scale: Current scales
        w_bf16_updated: Updated weights from optimizer step
        
    Returns:
        new_w_int8: Re-quantized weights
        new_scale: Updated scales
    """
    return quantize_int8(w_bf16_updated)


def quantize_int8_momentum(m_bf16: mx.array) -> Tuple[mx.array, mx.array]:
    """
    Quantize optimizer momentum to INT8 with per-tensor scaling.
    
    Uses per-tensor (not per-channel) scaling for optimizer states
    to reduce memory overhead from storing many scales.
    
    Args:
        m_bf16: Momentum tensor in bfloat16
        
    Returns:
        m_int8: Quantized momentum in int8
        scale: Scalar scale factor
    """
    abs_max = mx.max(mx.abs(m_bf16))
    scale = abs_max / 127.0 + 1e-8
    
    m_scaled = m_bf16 / scale
    m_int8 = mx.clip(mx.round(m_scaled), -127, 127).astype(mx.int8)
    
    return m_int8, scale.astype(mx.bfloat16)


def dequantize_int8_momentum(m_int8: mx.array, scale: mx.array) -> mx.array:
    """
    Dequantize INT8 momentum back to BF16.
    
    Args:
        m_int8: Quantized momentum in int8
        scale: Scalar scale factor
        
    Returns:
        m_bf16: Dequantized momentum in bfloat16
    """
    return m_int8.astype(mx.bfloat16) * scale

