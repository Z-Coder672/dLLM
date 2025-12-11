"""
8-bit AdamW optimizer for memory-efficient training.

- First moment (m): INT8 with per-tensor scaling
- Second moment (v): BF16 (needs precision for stability)
- Updates computed in BF16
"""

import mlx.core as mx
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import math

from .quantization import quantize_int8_momentum, dequantize_int8_momentum


@dataclass
class OptimizerState:
    """State for a single parameter."""
    m_int8: mx.array          # First moment (INT8)
    m_scale: mx.array         # Scale for first moment
    v: mx.array               # Second moment (BF16)
    step: int                 # Step count for bias correction


class AdamW8bit:
    """
    8-bit AdamW optimizer.
    
    Stores first moment in INT8 to save memory.
    Second moment kept in BF16 for stability.
    """
    
    def __init__(
        self,
        learning_rate: float = 3e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
    ):
        self.lr = learning_rate
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        self.state: Dict[int, OptimizerState] = {}
        self._step_count = 0
    
    def init_state(self, param: mx.array, param_id: int) -> OptimizerState:
        """Initialize optimizer state for a parameter."""
        # Initialize moments to zero
        m = mx.zeros_like(param, dtype=mx.bfloat16)
        v = mx.zeros_like(param, dtype=mx.bfloat16)
        
        # Quantize first moment to INT8
        m_int8, m_scale = quantize_int8_momentum(m)
        
        return OptimizerState(
            m_int8=m_int8,
            m_scale=m_scale,
            v=v,
            step=0,
        )
    
    def step(
        self,
        params: Dict[str, mx.array],
        grads: Dict[str, mx.array],
        lr: Optional[float] = None,
    ) -> Dict[str, mx.array]:
        """
        Perform a single optimization step.
        
        Args:
            params: Dictionary of parameter name -> parameter
            grads: Dictionary of parameter name -> gradient
            lr: Optional learning rate override
            
        Returns:
            Updated parameters
        """
        lr = lr if lr is not None else self.lr
        self._step_count += 1
        
        updated_params = {}
        
        for name, param in params.items():
            if name not in grads:
                updated_params[name] = param
                continue
            
            grad = grads[name]
            param_id = id(param)
            
            # Initialize state if needed
            if param_id not in self.state:
                self.state[param_id] = self.init_state(param, param_id)
            
            state = self.state[param_id]
            state.step += 1
            
            # Convert param to BF16 for computation
            param_bf16 = param.astype(mx.bfloat16)
            grad_bf16 = grad.astype(mx.bfloat16)
            
            # Dequantize first moment
            m = dequantize_int8_momentum(state.m_int8, state.m_scale)
            v = state.v
            
            # Update biased first moment
            m = self.beta1 * m + (1 - self.beta1) * grad_bf16
            
            # Update biased second moment
            v = self.beta2 * v + (1 - self.beta2) * (grad_bf16 * grad_bf16)
            
            # Bias correction
            bias_correction1 = 1 - self.beta1 ** state.step
            bias_correction2 = 1 - self.beta2 ** state.step
            
            m_hat = m / bias_correction1
            v_hat = v / bias_correction2
            
            # Compute update
            # Use float32 for numerical stability in division
            denom = mx.sqrt(v_hat.astype(mx.float32)) + self.eps
            update = m_hat.astype(mx.float32) / denom
            
            # Apply weight decay (decoupled)
            if self.weight_decay > 0:
                update = update + self.weight_decay * param_bf16.astype(mx.float32)
            
            # Update parameter
            new_param = param_bf16.astype(mx.float32) - lr * update
            new_param = new_param.astype(mx.bfloat16)
            
            # Quantize first moment back to INT8
            state.m_int8, state.m_scale = quantize_int8_momentum(m)
            state.v = v
            
            updated_params[name] = new_param
        
        return updated_params
    
    @property
    def step_count(self) -> int:
        return self._step_count
    
    def state_dict(self) -> Dict[str, Any]:
        """Get optimizer state for checkpointing."""
        return {
            "step_count": self._step_count,
            "lr": self.lr,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "state": {
                str(k): {
                    "m_int8": v.m_int8,
                    "m_scale": v.m_scale,
                    "v": v.v,
                    "step": v.step,
                }
                for k, v in self.state.items()
            }
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load optimizer state from checkpoint."""
        self._step_count = state_dict["step_count"]
        self.lr = state_dict["lr"]
        self.beta1 = state_dict["beta1"]
        self.beta2 = state_dict["beta2"]
        self.eps = state_dict["eps"]
        self.weight_decay = state_dict["weight_decay"]
        
        # Note: State reconstruction requires matching param IDs
        # This is handled during checkpoint loading


class SGDMomentum8bit:
    """
    Simple SGD with momentum, using INT8 for momentum storage.
    
    Even more memory efficient than AdamW8bit.
    """
    
    def __init__(
        self,
        learning_rate: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.1,
    ):
        self.lr = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        
        self.velocity: Dict[int, Tuple[mx.array, mx.array]] = {}  # (v_int8, scale)
        self._step_count = 0
    
    def step(
        self,
        params: Dict[str, mx.array],
        grads: Dict[str, mx.array],
        lr: Optional[float] = None,
    ) -> Dict[str, mx.array]:
        """Perform optimization step."""
        lr = lr if lr is not None else self.lr
        self._step_count += 1
        
        updated_params = {}
        
        for name, param in params.items():
            if name not in grads:
                updated_params[name] = param
                continue
            
            grad = grads[name]
            param_id = id(param)
            
            param_bf16 = param.astype(mx.bfloat16)
            grad_bf16 = grad.astype(mx.bfloat16)
            
            # Apply weight decay
            if self.weight_decay > 0:
                grad_bf16 = grad_bf16 + self.weight_decay * param_bf16
            
            # Get or initialize velocity
            if param_id in self.velocity:
                v_int8, v_scale = self.velocity[param_id]
                v = dequantize_int8_momentum(v_int8, v_scale)
            else:
                v = mx.zeros_like(param_bf16)
            
            # Update velocity
            v = self.momentum * v + grad_bf16
            
            # Update parameter
            new_param = param_bf16 - lr * v
            
            # Quantize velocity
            v_int8, v_scale = quantize_int8_momentum(v)
            self.velocity[param_id] = (v_int8, v_scale)
            
            updated_params[name] = new_param
        
        return updated_params


def get_cosine_schedule_with_warmup(
    step: int,
    warmup_steps: int,
    total_steps: int,
    min_lr: float,
    max_lr: float,
) -> float:
    """
    Cosine learning rate schedule with linear warmup.
    
    Args:
        step: Current step
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        min_lr: Minimum learning rate
        max_lr: Maximum learning rate
        
    Returns:
        Learning rate for current step
    """
    if step < warmup_steps:
        # Linear warmup
        return max_lr * step / warmup_steps
    
    # Cosine decay
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    
    return min_lr + (max_lr - min_lr) * cosine_decay


def clip_gradients(
    grads: Dict[str, mx.array],
    max_norm: float,
) -> Tuple[Dict[str, mx.array], float]:
    """
    Clip gradients by global norm.
    
    Args:
        grads: Dictionary of gradients
        max_norm: Maximum gradient norm
        
    Returns:
        Clipped gradients and the original norm
    """
    # Compute global norm
    total_norm_sq = mx.array(0.0, dtype=mx.float32)
    for grad in grads.values():
        total_norm_sq = total_norm_sq + mx.sum(grad.astype(mx.float32) ** 2)
    
    total_norm = mx.sqrt(total_norm_sq)
    
    # Clip if necessary
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef = mx.minimum(clip_coef, mx.array(1.0))
    
    clipped_grads = {
        name: grad * clip_coef.astype(grad.dtype)
        for name, grad in grads.items()
    }
    
    return clipped_grads, total_norm.item()

