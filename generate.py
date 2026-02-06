#!/usr/bin/env python3
"""
Text generation script for Ternary Transformer.

Usage:
    python generate.py --checkpoint checkpoints/500m/final --prompt "Once upon a time"
    python generate.py --checkpoint checkpoints/500m/final --interactive
    python generate.py --checkpoint checkpoints/500m/final --prompt "Test" --temperature 0.8 --top-p 0.9
"""

import argparse
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from src.config import ModelConfig
from src.model import create_model
from src.checkpoint import load_checkpoint
from data.dataloader import get_tokenizer


def sample_top_p(logits: mx.array, top_p: float) -> mx.array:
    """
    Nucleus (top-p) sampling - FAST MLX version.
    
    Args:
        logits: Shape (vocab_size,) - logits for next token
        top_p: Cumulative probability threshold
        
    Returns:
        Filtered logits with low-probability tokens set to -inf
    """
    # Convert to probabilities
    probs = mx.softmax(logits, axis=-1)
    
    # Sort probabilities in descending order
    sorted_indices = mx.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    
    # Cumulative sum
    cumsum_probs = mx.cumsum(sorted_probs, axis=-1)
    
    # Find cutoff: keep tokens until cumsum exceeds top_p
    # But always keep at least the first token
    cutoff_idx = mx.argmax((cumsum_probs > top_p).astype(mx.int32))
    if cutoff_idx.item() == 0:
        # If first token already exceeds top_p, keep it anyway
        cutoff_idx = mx.array(1)
    
    # Create mask for tokens to keep
    keep_mask = mx.arange(len(probs)) <= cutoff_idx
    
    # Apply mask in sorted order
    sorted_logits = logits[sorted_indices]
    sorted_logits = mx.where(keep_mask, sorted_logits, -float('inf'))
    
    # Scatter back to original order using fancy indexing
    # This is the key: we need to unsort the filtered logits
    result = mx.zeros_like(logits) - float('inf')
    # MLX supports scatter through index assignment in a roundabout way
    # We'll use the fact that sorted_indices tells us where each position goes
    scatter_indices = mx.argsort(sorted_indices)  # Inverse permutation
    result = sorted_logits[scatter_indices]
    
    return result


def sample_top_k(logits: mx.array, top_k: int) -> mx.array:
    """
    Top-k sampling - FAST MLX version.
    
    Args:
        logits: Shape (vocab_size,) - logits for next token
        top_k: Number of top tokens to keep
        
    Returns:
        Filtered logits with only top-k tokens, rest set to -inf
    """
    # Get top-k values
    top_k_values = mx.topk(logits, k=top_k)[0]
    threshold = top_k_values[-1]  # Smallest value in top-k
    
    # Keep only tokens >= threshold
    filtered_logits = mx.where(logits >= threshold, logits, -float('inf'))
    
    return filtered_logits


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
) -> str:
    """
    Generate text from a prompt.
    
    Args:
        model: The trained model
        tokenizer: Tokenizer
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (1.0 = unmodified, <1 = more focused, >1 = more random)
        top_p: Nucleus sampling threshold (1.0 = disabled, 0.9 = keep top 90% probability mass)
        top_k: Top-k sampling (0 = disabled, e.g. 50 = only sample from top 50 tokens)
        
    Returns:
        Generated text (WITHOUT the prompt)
    """
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt)
    prompt_length = len(input_ids)
    input_ids = mx.array([input_ids])
    
    # Generate tokens one at a time
    for _ in range(max_tokens):
        # Get logits for next token
        logits, _ = model(input_ids, training=False)
        next_token_logits = logits[0, -1, :]  # Shape: (vocab_size,)
        
        # Apply temperature scaling
        if temperature > 0:
            next_token_logits = next_token_logits / temperature
        
        # Apply top-k filtering if enabled
        if top_k > 0:
            next_token_logits = sample_top_k(next_token_logits, top_k)
        
        # Apply top-p (nucleus) filtering if enabled
        if top_p < 1.0:
            next_token_logits = sample_top_p(next_token_logits, top_p)
        
        # Convert to probabilities
        probs = mx.softmax(next_token_logits, axis=-1)
        
        # Sample from distribution or use greedy
        if temperature == 0:
            # Greedy decoding
            next_token = mx.argmax(probs)
        else:
            # Sample from distribution
            log_probs = mx.log(probs + 1e-10)
            next_token = mx.random.categorical(log_probs)
        
        # Reshape and append to sequence
        next_token = mx.array([[next_token.item()]])
        input_ids = mx.concatenate([input_ids, next_token], axis=1)
        
        # Force evaluation to free memory
        mx.eval(input_ids)
    
    # Decode only the newly generated tokens
    output_ids = input_ids[0].tolist()
    new_tokens = output_ids[prompt_length:]
    text = tokenizer.decode(new_tokens)
    
    return text


def interactive_mode(model, tokenizer, args):
    """Run interactive generation."""
    print("\n" + "=" * 60)
    print("Interactive Text Generation")
    print(f"Temperature: {args.temperature}, Top-p: {args.top_p}, Top-k: {args.top_k}")
    print("Type your prompt and press Enter. Type 'quit' to exit.")
    print("=" * 60 + "\n")
    
    while True:
        try:
            prompt = input("Prompt: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not prompt:
                continue
            
            print("\nGenerating...")
            print("-" * 40)
            
            text = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )
            
            # Display prompt + generation together
            print(f"{prompt}{text}")
            print("-" * 40 + "\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


def main():
    parser = argparse.ArgumentParser(description="Generate text with Ternary Transformer")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint directory")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt for generation")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode")
    parser.add_argument("--max-tokens", type=int, default=100,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (default: 0.8, use 0.0 for greedy)")
    parser.add_argument("--top-p", type=float, default=0.95,
                        help="Nucleus sampling threshold (default: 0.95, use 1.0 to disable)")
    parser.add_argument("--top-k", type=int, default=0,
                        help="Top-k sampling (default: 0/disabled)")
    args = parser.parse_args()
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    
    # Load config first
    state = load_checkpoint(args.checkpoint)
    config = state.get("config")
    
    if config is None:
        print("Error: Could not load model config from checkpoint")
        return
    
    print(f"Model: {config.d_model}d, {config.n_layers}L, {config.n_heads}H")
    
    # Create model
    model = create_model(config)
    
    # Load weights
    load_checkpoint(args.checkpoint, model=model)
    print("Model loaded successfully!")
    
    # Get tokenizer
    tokenizer = get_tokenizer()
    
    # Generate
    if args.interactive:
        interactive_mode(model, tokenizer, args)
    elif args.prompt:
        print(f"\nPrompt: {args.prompt}")
        print(f"Settings: temp={args.temperature}, top_p={args.top_p}, top_k={args.top_k}")
        print("\nGenerating...")
        print("-" * 40)
        
        text = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
        
        # Display prompt + generation together
        print(f"{args.prompt}{text}")
        print("-" * 40)
    else:
        print("Error: Provide --prompt or --interactive")


if __name__ == "__main__":
    main()