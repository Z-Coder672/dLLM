"""
Core layers for the Ternary Transformer.

- TernaryLinear: Linear layer with INT8 shadow weights and ternary forward
- RMSNorm: Root Mean Square Layer Normalization
- SwiGLU: Gated Linear Unit with Swish activation
"""

import mlx.core as mx
import mlx.nn as nn
import math
from typing import Optional

from .quantization import quantize_int8, dequantize_int8
from .ste import ternarize


class TernaryLinear(nn.Module):
    """
    Linear layer with INT8 shadow weights and ternary forward pass.
    
    Storage: INT8 weights (1 byte per param) + BF16 scales
    Forward: Dequantize -> Ternarize -> MatMul
    Backward: Gradients flow through STE to BF16, then requantize to INT8
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        threshold_factor: float = 0.7,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.threshold_factor = threshold_factor
        
        # Initialize in BF16, then quantize to INT8
        # Using scaled initialization for better training stability
        scale = 1.0 / math.sqrt(in_features)
        w_init = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(out_features, in_features),
            dtype=mx.bfloat16
        )
        
        # Quantize initial weights
        self._weight_int8, self._scale = quantize_int8(w_init)
        
        # Optional bias (kept in BF16)
        if bias:
            self._bias = mx.zeros((out_features,), dtype=mx.bfloat16)
        else:
            self._bias = None
    
    @property
    def weight_int8(self) -> mx.array:
        return self._weight_int8
    
    @property
    def scale(self) -> mx.array:
        return self._scale
    
    @property
    def bias(self) -> Optional[mx.array]:
        return self._bias
    
    def get_weight_bf16(self) -> mx.array:
        """Get dequantized weights in BF16."""
        return dequantize_int8(self._weight_int8, self._scale)
    
    def set_weight_bf16(self, w_bf16: mx.array):
        """Set weights from BF16 (re-quantizes to INT8)."""
        self._weight_int8, self._scale = quantize_int8(w_bf16)
    
    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass with ternary weights.
        
        Args:
            x: Input tensor, shape (..., in_features)
            
        Returns:
            Output tensor, shape (..., out_features)
        """
        # Dequantize INT8 -> BF16
        w_bf16 = dequantize_int8(self._weight_int8, self._scale)
        
        # Ternarize for forward pass (STE handles backward)
        w_ternary = ternarize(w_bf16, self.threshold_factor)
        
        # Matrix multiplication
        # x: (..., in_features), w_ternary: (out_features, in_features)
        # Result: (..., out_features)
        out = x @ w_ternary.T
        
        if self._bias is not None:
            out = out + self._bias
        
        return out
    
    def update_from_gradient(self, grad: mx.array, lr: float):
        """
        Update weights from gradient (for manual optimization).
        
        This is a simplified update; the full optimizer handles this.
        """
        w_bf16 = self.get_weight_bf16()
        w_bf16 = w_bf16 - lr * grad
        self.set_weight_bf16(w_bf16)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    More efficient than LayerNorm (no mean subtraction).
    output = input * rsqrt(mean(input²) + eps) * scale
    """
    
    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dims,), dtype=mx.bfloat16)
    
    def __call__(self, x: mx.array) -> mx.array:
        # Compute RMS
        # Use float32 for numerical stability in the norm computation
        x_f32 = x.astype(mx.float32)
        rms = mx.sqrt(mx.mean(x_f32 * x_f32, axis=-1, keepdims=True) + self.eps)
        
        # Normalize and scale
        x_norm = (x_f32 / rms).astype(mx.bfloat16)
        return x_norm * self.weight


class SwiGLU(nn.Module):
    """
    SwiGLU activation for FFN.
    
    SwiGLU(x) = Swish(xW_gate) * (xW_up)
    
    Uses ternary linear layers for both projections.
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        threshold_factor: float = 0.7,
    ):
        super().__init__()
        
        # Gate and up projections
        self.w_gate = TernaryLinear(d_model, d_ff, threshold_factor=threshold_factor)
        self.w_up = TernaryLinear(d_model, d_ff, threshold_factor=threshold_factor)
        
        # Down projection
        self.w_down = TernaryLinear(d_ff, d_model, threshold_factor=threshold_factor)
    
    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: Input tensor, shape (..., d_model)
            
        Returns:
            Output tensor, shape (..., d_model)
        """
        # Swish(gate) * up
        gate = mx.sigmoid(self.w_gate(x)) * self.w_gate(x)  # Swish = x * sigmoid(x)
        up = self.w_up(x)
        hidden = gate * up
        
        # Down projection
        return self.w_down(hidden)


class FeedForward(nn.Module):
    """
    Feed-forward network with SwiGLU activation.
    
    This is the standard FFN block used in each transformer layer.
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        threshold_factor: float = 0.7,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.swiglu = SwiGLU(d_model, d_ff, threshold_factor)
        self.dropout = dropout
    
    def __call__(self, x: mx.array, training: bool = False) -> mx.array:
        out = self.swiglu(x)
        
        if training and self.dropout > 0:
            out = mx.dropout(out, p=self.dropout)
        
        return out


class Embedding(nn.Module):
    """
    Token embedding layer.
    
    Kept in BF16 (not quantized) for better gradient signal.
    """
    
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        # Initialize with small values
        scale = 1.0 / math.sqrt(d_model)
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(vocab_size, d_model),
            dtype=mx.bfloat16
        )
    
    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: Token indices, shape (...,)
            
        Returns:
            Embeddings, shape (..., d_model)
        """
        return self.weight[x]

