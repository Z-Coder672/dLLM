"""
Straight-Through Estimator (STE) for ternary quantization.

The forward pass uses ternary weights {-1, 0, +1}.
The backward pass passes gradients through unchanged (identity).
"""

import mlx.core as mx


def _ternarize_impl(w: mx.array, threshold_factor: float) -> mx.array:
    """
    Core ternarization logic.
    
    Args:
        w: Weights in bfloat16
        threshold_factor: Factor to compute threshold as τ = factor * mean(|W|)
        
    Returns:
        Ternary weights {-1, 0, +1} in bfloat16
    """
    # Compute adaptive threshold based on weight magnitude
    threshold = threshold_factor * mx.mean(mx.abs(w))
    
    # Ternarize: values above threshold become sign(w), others become 0
    ternary = mx.where(
        mx.abs(w) > threshold,
        mx.sign(w),
        mx.zeros_like(w)
    )
    
    return ternary


def ternarize(w: mx.array, threshold_factor: float = 0.7) -> mx.array:
    """
    Ternarize weights with Straight-Through Estimator.
    
    Forward: w -> {-1, 0, +1} based on threshold
    Backward: Gradient passes through unchanged
    
    Args:
        w: Weights in bfloat16
        threshold_factor: Factor to compute threshold as τ = factor * mean(|W|)
        
    Returns:
        Ternary weights {-1, 0, +1} in bfloat16
    """
    return _ternarize_ste(w, threshold_factor)


@mx.custom_function
def _ternarize_ste(w: mx.array, threshold_factor: float) -> mx.array:
    """
    Ternarization with custom VJP for straight-through gradient.
    """
    return _ternarize_impl(w, threshold_factor)


@_ternarize_ste.vjp
def _ternarize_vjp(primals, cotangents, output):
    """
    VJP (Vector-Jacobian Product) for ternarization.
    
    Implements Straight-Through Estimator:
    - Forward: ternarize weights
    - Backward: pass gradient through unchanged (identity Jacobian)
    
    Args:
        primals: (w, threshold_factor) - original inputs
        cotangents: (grad_output,) - gradient from downstream
        output: ternarized weights (unused)
        
    Returns:
        Tuple of gradients for each input (grad_w, None for threshold_factor)
    """
    grad_output = cotangents
    # STE: gradient passes through unchanged
    # No gradient for threshold_factor (it's a hyperparameter)
    return grad_output, None


def ternarize_stochastic(w: mx.array, threshold_factor: float = 0.7, 
                         noise_scale: float = 0.1, key: mx.array = None) -> mx.array:
    """
    Stochastic ternarization for training exploration.
    
    Adds small noise before ternarization to encourage exploration
    of different ternary configurations during training.
    
    Args:
        w: Weights in bfloat16
        threshold_factor: Factor to compute threshold
        noise_scale: Scale of uniform noise to add
        key: Random key for noise generation
        
    Returns:
        Ternary weights {-1, 0, +1} in bfloat16
    """
    if key is None:
        key = mx.random.key(0)
    
    # Add small uniform noise
    noise = mx.random.uniform(
        low=-noise_scale, 
        high=noise_scale, 
        shape=w.shape, 
        key=key,
        dtype=w.dtype
    )
    w_noisy = w + noise * mx.std(w)
    
    return ternarize(w_noisy, threshold_factor)


def compute_ternary_stats(w: mx.array, threshold_factor: float = 0.7) -> dict:
    """
    Compute statistics about the ternary weight distribution.
    
    Useful for monitoring training health.
    
    Args:
        w: Weights in bfloat16
        threshold_factor: Threshold factor used for ternarization
        
    Returns:
        Dict with counts and percentages of {-1, 0, +1}
    """
    ternary = _ternarize_impl(w, threshold_factor)
    total = ternary.size
    
    n_pos = mx.sum(ternary > 0).item()
    n_neg = mx.sum(ternary < 0).item()
    n_zero = mx.sum(ternary == 0).item()
    
    return {
        "n_positive": n_pos,
        "n_negative": n_neg, 
        "n_zero": n_zero,
        "pct_positive": 100 * n_pos / total,
        "pct_negative": 100 * n_neg / total,
        "pct_zero": 100 * n_zero / total,
        "total": total,
    }

