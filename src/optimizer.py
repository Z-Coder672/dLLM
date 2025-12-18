"""
AdamW optimizer tuned for ternary training stability.

- Moments kept in float32 for numerical safety
- Supports BF16 parameters and updates
"""

import mlx.core as mx
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import math


@dataclass
class OptimizerState:
    """State for a single parameter."""
    m: mx.array               # First moment (FP32)
    v: mx.array               # Second moment (FP32)
    step: int                 # Step count for bias correction


class AdamW:
    """
    AdamW optimizer for BF16 weights.
    
    First and second moments kept in float32 for stability.
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
        
        self.state: Dict[str, OptimizerState] = {}
        self._step_count = 0
    
    def init_state(self, param: mx.array) -> OptimizerState:
        """Initialize optimizer state for a parameter."""
        # Keep optimizer state in float32 for stability
        m = mx.zeros(param.shape, dtype=mx.float32)
        v = mx.zeros(param.shape, dtype=mx.float32)
        return OptimizerState(m=m, v=v, step=0)
    
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
            param_key = name
            
            # Initialize state if needed
            if param_key not in self.state:
                self.state[param_key] = self.init_state(param)
            
            state = self.state[param_key]
            state.step += 1
            
            # Convert inputs to float32 for stable moment updates
            param_f32 = param.astype(mx.float32)
            grad_f32 = grad.astype(mx.float32)
            
            # Get moments
            m = state.m
            v = state.v
            
            if m.shape != grad_f32.shape:
                raise ValueError(
                    f"Shape mismatch for {name}: m {m.shape}, grad {grad.shape}, param {param.shape}"
                )
            
            # Update biased first moment
            m = self.beta1 * m + (1 - self.beta1) * grad_f32
            
            # Update biased second moment
            v = self.beta2 * v + (1 - self.beta2) * (grad_f32 * grad_f32)
            
            # Bias correction
            bias_correction1 = 1 - self.beta1 ** state.step
            bias_correction2 = 1 - self.beta2 ** state.step
            
            m_hat = m / bias_correction1
            v_hat = v / bias_correction2
            
            # Compute update
            # Use float32 for numerical stability in division
            denom = mx.sqrt(v_hat) + self.eps
            update = m_hat / denom
            
            # Apply weight decay (decoupled)
            if self.weight_decay > 0:
                update = update + self.weight_decay * param_f32
            
            # Update parameter
            new_param = param_f32 - lr * update
            new_param = new_param.astype(param.dtype)
            
            # Update state
            state.m = m
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
                    "m": v.m,
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


class SGDMomentum:
    """
    Simple SGD with momentum for BF16 weights.
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
        
        self.velocity: Dict[str, mx.array] = {}  # velocity in BF16
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
            param_key = name
            
            param_bf16 = param.astype(mx.bfloat16)
            grad_bf16 = grad.astype(mx.bfloat16)
            
            # Apply weight decay
            if self.weight_decay > 0:
                grad_bf16 = grad_bf16 + self.weight_decay * param_bf16
            
            # Get or initialize velocity
            if param_key in self.velocity:
                v = self.velocity[param_key]
            else:
                v = mx.zeros_like(param_bf16)
            
            # Update velocity
            v = self.momentum * v + grad_bf16
            
            # Update parameter
            new_param = param_bf16 - lr * v
            
            # Store velocity
            self.velocity[param_key] = v
            
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

