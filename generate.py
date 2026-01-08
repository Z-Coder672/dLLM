#!/usr/bin/env python3
"""
Text generation script for Ternary Transformer.

Usage:
    python generate.py --checkpoint checkpoints/500m/final --prompt "Once upon a time"
    python generate.py --checkpoint checkpoints/500m/final --interactive
"""

import argparse
from pathlib import Path

import mlx.core as mx

from src.config import ModelConfig
from src.model import create_model
from src.checkpoint import load_checkpoint
from data.dataloader import get_tokenizer


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 1.0,
) -> str:
    """
    Generate text from a prompt using simple sampling.
    
    Args:
        model: The trained model
        tokenizer: Tokenizer
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (1.0 = unmodified, <1 = focused, >1 = random)
        
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
        next_token_logits = logits[:, -1, :]  # Shape: (1, vocab_size)
        
        # Apply temperature
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature
        
        # Sample from distribution
        probs = mx.softmax(next_token_logits, axis=-1)
        next_token = mx.random.categorical(mx.log(probs + 1e-10))
        next_token = next_token[:, None]  # Add sequence dimension
        
        # Append to sequence
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
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (lower=more focused, higher=more random)")
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
        print("\nGenerating...")
        print("-" * 40)
        
        text = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        
        # Display prompt + generation together
        print(f"{args.prompt}{text}")
        print("-" * 40)
    else:
        print("Error: Provide --prompt or --interactive")


if __name__ == "__main__":
    main()