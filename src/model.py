"""
Full Ternary Transformer model.

Implements a decoder-only transformer with:
- INT8 shadow weights + ternary forward pass
- RMSNorm pre-normalization
- RoPE positional embeddings
- SwiGLU FFN
- Gradient checkpointing support
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple, List
from functools import partial

from .config import ModelConfig
from .layers import TernaryLinear, RMSNorm, FeedForward, Embedding
from .attention import MultiHeadAttention, create_causal_mask


class TransformerBlock(nn.Module):
    """
    Single transformer block with pre-norm architecture.
    
    Architecture:
        x -> RMSNorm -> Attention -> + -> RMSNorm -> FFN -> +
            └────────────────────────┘    └───────────────┘
    """
    
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        
        self.layer_idx = layer_idx
        
        # Pre-attention norm
        self.attn_norm = RMSNorm(config.d_model)
        
        # Multi-head attention
        self.attention = MultiHeadAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            threshold_factor=config.threshold_factor,
            max_seq_len=config.max_seq_len,
            rope_theta=config.rope_theta,
            dropout=config.dropout,
        )
        
        # Pre-FFN norm
        self.ffn_norm = RMSNorm(config.d_model)
        
        # Feed-forward network
        self.ffn = FeedForward(
            d_model=config.d_model,
            d_ff=config.d_ff,
            threshold_factor=config.threshold_factor,
            dropout=config.dropout,
        )
    
    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
        training: bool = False,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        """
        Forward pass through the transformer block.
        
        Args:
            x: Input tensor, shape (batch, seq_len, d_model)
            mask: Causal attention mask
            cache: Optional KV cache
            training: Whether in training mode
            
        Returns:
            output: Output tensor, same shape as input
            new_cache: Updated KV cache
        """
        # Attention with residual
        h = self.attn_norm(x)
        attn_out, new_cache = self.attention(h, mask, cache, training)
        x = x + attn_out
        
        # FFN with residual
        h = self.ffn_norm(x)
        ffn_out = self.ffn(h, training)
        x = x + ffn_out
        
        return x, new_cache


class TernaryTransformer(nn.Module):
    """
    Full Ternary Transformer language model.
    
    Components:
    - Token embeddings (BF16, not quantized)
    - N transformer blocks with ternary weights
    - Final RMSNorm
    - Output projection (tied with embeddings or separate)
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.config = config
        
        # Token embeddings (kept in BF16)
        self.embed = Embedding(config.vocab_size, config.d_model)
        
        # Transformer blocks
        self.layers = []
        for i in range(config.n_layers):
            layer = TransformerBlock(config, layer_idx=i)
            # Register each layer as an attribute so named_modules() can find it
            setattr(self, f"layer_{i}", layer)
            self.layers.append(layer)
        
        # Final normalization
        self.norm = RMSNorm(config.d_model)
        
        # Output projection (can tie weights with embeddings)
        # Using ternary for output projection
        self.lm_head = TernaryLinear(
            config.d_model,
            config.vocab_size,
            threshold_factor=config.threshold_factor,
        )
    
    def __call__(
        self,
        input_ids: mx.array,
        cache: Optional[List[Tuple[mx.array, mx.array]]] = None,
        training: bool = False,
    ) -> Tuple[mx.array, Optional[List[Tuple[mx.array, mx.array]]]]:
        """
        Forward pass through the transformer.
        
        Args:
            input_ids: Token indices, shape (batch, seq_len)
            cache: Optional list of KV caches for each layer
            training: Whether in training mode
            
        Returns:
            logits: Output logits, shape (batch, seq_len, vocab_size)
            new_cache: Updated KV caches
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        x = self.embed(input_ids)
        
        # Create causal mask
        mask = create_causal_mask(seq_len, x.dtype)
        
        # Initialize cache list if needed
        new_cache = [] if cache is not None else None
        
        # Pass through transformer blocks
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            x, updated_cache = layer(x, mask, layer_cache, training)
            
            if new_cache is not None:
                new_cache.append(updated_cache)
        
        # Final normalization
        x = self.norm(x)
        
        # Output projection to vocabulary
        logits = self.lm_head(x)
        
        return logits, new_cache
    
    def forward_with_checkpointing(
        self,
        input_ids: mx.array,
        checkpoint_every: int = 6,
        training: bool = True,
    ) -> mx.array:
        """
        Forward pass with gradient checkpointing.
        
        Checkpoints activations every N layers to reduce memory usage.
        Only used during training (no cache support).
        
        Args:
            input_ids: Token indices, shape (batch, seq_len)
            checkpoint_every: Checkpoint every N layers
            training: Should be True for checkpointing
            
        Returns:
            logits: Output logits
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        x = self.embed(input_ids)
        
        # Create causal mask
        mask = create_causal_mask(seq_len, x.dtype)
        
        # Process layers with checkpointing
        for i, layer in enumerate(self.layers):
            if i > 0 and i % checkpoint_every == 0:
                # Use checkpoint - recompute forward in backward pass
                x = mx.checkpoint(partial(self._layer_forward, layer, mask, training))(x)
            else:
                x, _ = layer(x, mask, None, training)
        
        # Final normalization
        x = self.norm(x)
        
        # Output projection
        logits = self.lm_head(x)
        
        return logits
    
    def _layer_forward(
        self,
        layer: TransformerBlock,
        mask: mx.array,
        training: bool,
        x: mx.array
    ) -> mx.array:
        """Helper for checkpointing a single layer."""
        out, _ = layer(x, mask, None, training)
        return out
    
    def generate(
        self,
        input_ids: mx.array,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> mx.array:
        """
        Generate tokens autoregressively.
        
        Args:
            input_ids: Initial token indices, shape (batch, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: If set, only sample from top-k logits
            top_p: If set, use nucleus sampling
            
        Returns:
            Generated token sequence including input
        """
        # Initialize KV cache
        cache = None
        generated = input_ids
        
        for _ in range(max_new_tokens):
            # Forward pass (use full sequence first time, then just new token)
            if cache is None:
                logits, cache = self(generated, cache=None, training=False)
            else:
                # Only pass the last token when using cache
                logits, cache = self(generated[:, -1:], cache=cache, training=False)
            
            # Get logits for the last position
            next_logits = logits[:, -1, :]
            
            # Apply temperature
            if temperature != 1.0:
                next_logits = next_logits / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                top_k_logits, top_k_indices = mx.topk(next_logits, k=top_k)
                next_logits = mx.full_like(next_logits, float("-inf"))
                # Scatter top-k values back
                for i in range(next_logits.shape[0]):
                    next_logits[i, top_k_indices[i]] = top_k_logits[i]
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits = mx.sort(next_logits, axis=-1)[:, ::-1]
                sorted_probs = mx.softmax(sorted_logits, axis=-1)
                cumsum_probs = mx.cumsum(sorted_probs, axis=-1)
                
                # Remove tokens with cumulative prob above threshold
                sorted_mask = cumsum_probs > top_p
                # Keep at least one token
                sorted_mask = mx.concatenate([
                    mx.zeros_like(sorted_mask[:, :1]),
                    sorted_mask[:, :-1]
                ], axis=-1)
                
                sorted_logits = mx.where(sorted_mask, float("-inf"), sorted_logits)
                # Unsort (approximate - just use softmax on filtered)
                next_logits = sorted_logits
            
            # Sample from distribution
            probs = mx.softmax(next_logits.astype(mx.float32), axis=-1)
            next_token = mx.random.categorical(mx.log(probs + 1e-10))
            next_token = next_token[:, None]
            
            # Append to generated sequence
            generated = mx.concatenate([generated, next_token], axis=1)
            
            # Evaluate to free memory
            mx.eval(generated, cache)
        
        return generated
    
    def count_parameters(self) -> dict:
        """Count parameters by component."""
        total = 0
        breakdown = {}
        seen = set()

        # Traverse all modules and count their weights explicitly so we
        # include ternary layers that don't expose parameters via
        # module.parameters().
        for name, module in self.named_modules():
            if id(module) in seen:
                continue
            seen.add(id(module))

            prefix = f"{name}." if name else ""

            if isinstance(module, TernaryLinear):
                w_count = module.weight_int8.size
                total += w_count
                breakdown[f"{prefix}weight"] = w_count

                # Scales are small but stored alongside weights
                scale_count = module.scale.size
                total += scale_count
                breakdown[f"{prefix}scale"] = scale_count

                if module.bias is not None:
                    b_count = module.bias.size
                    total += b_count
                    breakdown[f"{prefix}bias"] = b_count
                continue

            weight = getattr(module, "weight", None)
            if isinstance(weight, mx.array):
                w_count = weight.size
                total += w_count
                breakdown[f"{prefix}weight"] = w_count

            bias = getattr(module, "bias", None)
            if isinstance(bias, mx.array):
                b_count = bias.size
                total += b_count
                breakdown[f"{prefix}bias"] = b_count

        breakdown["total"] = total
        return breakdown


def create_model(config: ModelConfig) -> TernaryTransformer:
    """Create a TernaryTransformer from config."""
    return TernaryTransformer(config)

