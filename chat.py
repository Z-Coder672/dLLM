#!/usr/bin/env python3
"""
Inference script for Ternary Transformer.

Usage:
    # Single prompt
    python chat.py --checkpoint checkpoints/500m/step_5800 --prompt "Once upon a time"

    # Interactive chat
    python chat.py --checkpoint checkpoints/500m/step_5800 --interactive

    # With sampling params
    python chat.py --checkpoint checkpoints/500m/step_5800 --prompt "Hello" \
        --temperature 0.7 --top-p 0.9 --top-k 50 --max-tokens 256
"""

import argparse
import sys
import time

import mlx.core as mx

from src.config import ModelConfig
from src.model import TernaryTransformer, create_model
from src.checkpoint import load_checkpoint
from data.dataloader import get_tokenizer


# GPT-2 end-of-text token
EOT_TOKEN = 50256


def load_model(checkpoint_path: str) -> TernaryTransformer:
    """Load model from a checkpoint directory."""
    state = load_checkpoint(checkpoint_path)
    config = state["config"]
    model = create_model(config)
    load_checkpoint(checkpoint_path, model=model)

    # Respect the training config's ternary schedule — if ternary was never
    # activated during training, don't enable it at inference either.
    training_config = state.get("training_config")
    step = state.get("step", 0)
    if training_config and step < training_config.ternary_skip_steps:
        model.set_ternary_enabled(False)
    else:
        model.set_ternary_enabled(True)
        model.set_ternary_strength(1.0)

    return model, config


def generate_streaming(
    model: TernaryTransformer,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 0.3,
    top_k: int = 40,
    top_p: float = 0.9,
):
    """Generate tokens one at a time, printing as they're produced."""
    # Print prompt in dim color, then generate in default color
    sys.stdout.write(f"\033[2m{prompt}\033[0m")
    sys.stdout.flush()

    input_ids = tokenizer.encode(prompt)
    tokens = mx.array([input_ids])
    cache = None

    # Prefill: run the full prompt through the model
    logits, cache = model(tokens, cache=None, training=False)
    mx.eval(logits, cache)

    generated_tokens = []

    for _ in range(max_new_tokens):
        # Sample from the last position's logits
        next_logits = logits[:, -1, :].astype(mx.float32)

        # Temperature
        if temperature > 0:
            next_logits = next_logits / temperature
        else:
            # Greedy
            next_token = mx.argmax(next_logits, axis=-1, keepdims=True)
            token_id = next_token.item()
            if token_id == EOT_TOKEN:
                break
            generated_tokens.append(token_id)
            sys.stdout.write(tokenizer.decode([token_id]))
            sys.stdout.flush()
            tokens = next_token
            logits, cache = model(tokens, cache=cache, training=False)
            mx.eval(logits, cache)
            continue

        # Top-k filtering
        if top_k > 0:
            k = min(top_k, next_logits.shape[-1])
            top_k_vals = mx.topk(next_logits, k=k)
            # Threshold: everything below the k-th largest gets -inf
            threshold = mx.min(top_k_vals, axis=-1, keepdims=True)
            next_logits = mx.where(next_logits < threshold, float("-inf"), next_logits)

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_indices = mx.argsort(next_logits, axis=-1)[:, ::-1]
            sorted_logits = mx.take_along_axis(next_logits, sorted_indices, axis=-1)
            sorted_probs = mx.softmax(sorted_logits, axis=-1)
            cumsum_probs = mx.cumsum(sorted_probs, axis=-1)

            # Mask tokens whose cumulative prob exceeds top_p (keep at least 1)
            sorted_mask = mx.concatenate(
                [mx.zeros_like(cumsum_probs[:, :1]), (cumsum_probs[:, :-1] >= top_p)],
                axis=-1,
            )
            sorted_logits = mx.where(sorted_mask, float("-inf"), sorted_logits)

            # Unsort back to original vocab order
            unsort_indices = mx.argsort(sorted_indices, axis=-1)
            next_logits = mx.take_along_axis(sorted_logits, unsort_indices, axis=-1)

        # Sample
        probs = mx.softmax(next_logits, axis=-1)
        next_token = mx.random.categorical(mx.log(probs + 1e-10))
        next_token = next_token[:, None]
        token_id = next_token.item()

        if token_id == EOT_TOKEN:
            break

        generated_tokens.append(token_id)
        sys.stdout.write(tokenizer.decode([token_id]))
        sys.stdout.flush()

        # Next step with KV cache
        tokens = next_token
        logits, cache = model(tokens, cache=cache, training=False)
        mx.eval(logits, cache)

    print()  # Final newline
    return generated_tokens


def interactive_mode(model, tokenizer, config, args):
    """Multi-turn interactive CLI chat."""
    max_ctx = config.max_seq_len
    print(f"Interactive mode (context window: {max_ctx} tokens). Type 'quit' or Ctrl-C to exit.\n")

    history_tokens = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Bye!")
            break

        new_tokens = tokenizer.encode(user_input)
        history_tokens.extend(new_tokens)

        # Truncate history to fit within context window (leave room for generation)
        max_gen = args.max_tokens if args.max_tokens else max_ctx
        max_prompt = max_ctx - min(max_gen, max_ctx // 2)
        if len(history_tokens) > max_prompt:
            history_tokens = history_tokens[-max_prompt:]

        print("Model: ", end="", flush=True)

        tokens = mx.array([history_tokens])
        cache = None
        logits, cache = model(tokens, cache=None, training=False)
        mx.eval(logits, cache)

        gen_count = 0
        gen_limit = min(max_gen, max_ctx - len(history_tokens))
        response_tokens = []

        for _ in range(gen_limit):
            next_logits = logits[:, -1, :].astype(mx.float32)

            if args.temperature > 0:
                next_logits = next_logits / args.temperature
            else:
                next_token = mx.argmax(next_logits, axis=-1, keepdims=True)
                token_id = next_token.item()
                if token_id == EOT_TOKEN:
                    break
                response_tokens.append(token_id)
                sys.stdout.write(tokenizer.decode([token_id]))
                sys.stdout.flush()
                logits, cache = model(next_token, cache=cache, training=False)
                mx.eval(logits, cache)
                gen_count += 1
                continue

            # Top-k
            if args.top_k > 0:
                k = min(args.top_k, next_logits.shape[-1])
                top_k_vals = mx.topk(next_logits, k=k)
                threshold = mx.min(top_k_vals, axis=-1, keepdims=True)
                next_logits = mx.where(next_logits < threshold, float("-inf"), next_logits)

            # Top-p
            if args.top_p < 1.0:
                sorted_indices = mx.argsort(next_logits, axis=-1)[:, ::-1]
                sorted_logits = mx.take_along_axis(next_logits, sorted_indices, axis=-1)
                sorted_probs = mx.softmax(sorted_logits, axis=-1)
                cumsum_probs = mx.cumsum(sorted_probs, axis=-1)
                sorted_mask = mx.concatenate(
                    [mx.zeros_like(cumsum_probs[:, :1]), (cumsum_probs[:, :-1] >= args.top_p)],
                    axis=-1,
                )
                sorted_logits = mx.where(sorted_mask, float("-inf"), sorted_logits)
                unsort_indices = mx.argsort(sorted_indices, axis=-1)
                next_logits = mx.take_along_axis(sorted_logits, unsort_indices, axis=-1)

            probs = mx.softmax(next_logits, axis=-1)
            next_token = mx.random.categorical(mx.log(probs + 1e-10))
            next_token = next_token[:, None]
            token_id = next_token.item()

            if token_id == EOT_TOKEN:
                break

            response_tokens.append(token_id)
            sys.stdout.write(tokenizer.decode([token_id]))
            sys.stdout.flush()

            logits, cache = model(next_token, cache=cache, training=False)
            mx.eval(logits, cache)
            gen_count += 1

        print()  # Newline after response
        history_tokens.extend(response_tokens)


def main():
    parser = argparse.ArgumentParser(description="Run inference with a trained Ternary Transformer")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint directory (e.g. checkpoints/500m/step_5800)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt for single-shot generation")
    parser.add_argument("--interactive", action="store_true",
                        help="Launch multi-turn interactive chat")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Sampling temperature (default: 0.3, 0 = greedy)")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Nucleus sampling threshold (default: 0.9)")
    parser.add_argument("--top-k", type=int, default=40,
                        help="Top-k filtering (default: 40, 0 = disabled)")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Max tokens to generate (default: fill context window)")
    args = parser.parse_args()

    if not args.prompt and not args.interactive:
        parser.error("Specify either --prompt or --interactive")

    # Load model
    print(f"Loading checkpoint from {args.checkpoint}...")
    model, config = load_model(args.checkpoint)

    total_params = model.count_parameters().get("total", 0)
    print(f"Model loaded: {total_params / 1e6:.1f}M parameters, context {config.max_seq_len} tokens")

    # Tokenizer
    tokenizer = get_tokenizer()

    # Default max tokens = full context window minus prompt
    if args.max_tokens is None:
        max_tokens = config.max_seq_len
    else:
        max_tokens = args.max_tokens

    if args.interactive:
        interactive_mode(model, tokenizer, config, args)
    else:
        start = time.time()
        gen_tokens = generate_streaming(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        elapsed = time.time() - start
        n = len(gen_tokens)
        if n > 0:
            print(f"\n[{n} tokens in {elapsed:.1f}s — {n / elapsed:.1f} tok/s]")


if __name__ == "__main__":
    main()
