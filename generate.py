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
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
) -> str:
    """
    Generate text from a prompt.
    
    Args:
        model: The trained model
        tokenizer: Tokenizer
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling parameter
        
    Returns:
        Generated text
    """
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt)
    input_ids = mx.array([input_ids])  # Add batch dimension
    
    # Generate
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    
    # Decode
    output_ids = output_ids[0].tolist()  # Remove batch dimension
    text = tokenizer.decode(output_ids)
    
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
            
            print("\nGenerating...\n")
            
            text = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )
            
            print("-" * 40)
            print(text)
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
    parser.add_argument("--max-tokens", type=int, default=200,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top-k sampling parameter")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Nucleus sampling parameter")
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
        print("\nGenerating...\n")
        
        text = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        
        print("-" * 40)
        print(text)
        print("-" * 40)
    else:
        print("Error: Provide --prompt or --interactive")


if __name__ == "__main__":
    main()

