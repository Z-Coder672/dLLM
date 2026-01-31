"""
Streaming data loader for text datasets on HuggingFace.

Features:
- Streaming from HuggingFace datasets (no full download)
- GPT-2 tokenizer
- Sequence packing for efficiency
- Shuffle buffer for randomization
"""

import os
from pathlib import Path

import mlx.core as mx
import numpy as np
from typing import Iterator, Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
import random
import tiktoken


def get_tokenizer():
    """Get GPT-2 tokenizer using tiktoken."""
    return tiktoken.get_encoding("gpt2")


@dataclass
class Batch:
    """A training batch."""
    input_ids: mx.array   # (batch_size, seq_len)
    labels: mx.array      # (batch_size, seq_len) - shifted by 1
    
    @property
    def batch_size(self) -> int:
        return self.input_ids.shape[0]
    
    @property
    def seq_len(self) -> int:
        return self.input_ids.shape[1]


class StreamingTextDataset:
    """
    Streaming dataset for arbitrary HuggingFace text datasets.
    
    Loads data with streaming enabled and packs text into fixed-length token sequences.
    """
    
    def __init__(
        self,
        split: str = "train",
        seq_len: int = 1024,
        shuffle_buffer: int = 10000,
        dataset_name: str = "wikitext",
        dataset_config: Optional[str] = "wikitext-103-raw-v1",
    ):
        self.split = split
        self.seq_len = seq_len
        self.shuffle_buffer = shuffle_buffer
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.tokenizer = get_tokenizer()
        self.cache_dir = (Path(__file__).resolve().parent / "cache").expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Special tokens
        self.eos_token = self.tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
        
        self._dataset = None
    
    def _dataset_cache_present(self) -> bool:
        """Check if the dataset is already cached locally."""
        cache_pattern = self.dataset_name.replace("/", "_")
        return any(path.is_dir() for path in self.cache_dir.glob(f"{cache_pattern}*"))
    
    def _load_dataset(self):
        """Lazily load the dataset."""
        if self._dataset is None:
            from datasets import DownloadConfig, load_dataset
            
            # Allow slower but more reliable downloads
            os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")
            os.environ.setdefault("HF_HUB_HTTP_TIMEOUT", "120")
            os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
            
            def _do_load(local_only: bool):
                download_config = DownloadConfig(
                    cache_dir=str(self.cache_dir),
                    max_retries=5,
                    local_files_only=local_only,
                )
                if self.dataset_config:
                    return load_dataset(
                        self.dataset_name,
                        self.dataset_config,
                        split=self.split,
                        streaming=True,
                        cache_dir=str(self.cache_dir),
                        download_config=download_config,
                    )
                return load_dataset(
                    self.dataset_name,
                    split=self.split,
                    streaming=True,
                    cache_dir=str(self.cache_dir),
                    download_config=download_config,
                )
            
            try:
                self._dataset = _do_load(local_only=self._dataset_cache_present())
            except Exception:
                # If cache-only load fails (or cache missing), fall back to download
                self._dataset = _do_load(local_only=False)
            if self.shuffle_buffer > 0:
                self._dataset = self._dataset.shuffle(
                    buffer_size=self.shuffle_buffer,
                    seed=42,
                )
    
    def _tokenize_story(self, text: str) -> List[int]:
        """Tokenize a single story."""
        tokens = self.tokenizer.encode(text)
        tokens.append(self.eos_token)  # Add EOS
        return tokens
    
    def __iter__(self) -> Iterator[List[int]]:
        """
        Iterate over packed sequences.
        
        Yields sequences of exactly seq_len tokens, packed from multiple stories.
        """
        self._load_dataset()
        
        buffer = []
        
        for example in self._dataset:
            text = example.get("text", "")
            if not text:
                continue
            
            tokens = self._tokenize_story(text)
            buffer.extend(tokens)
            
            # Yield complete sequences
            while len(buffer) >= self.seq_len + 1:  # +1 for labels
                sequence = buffer[:self.seq_len + 1]
                buffer = buffer[self.seq_len:]  # Keep remainder (overlapping by 1)
                yield sequence
    
    def get_batch_iterator(
        self,
        batch_size: int,
    ) -> Iterator[Batch]:
        """
        Get an iterator that yields batches.
        
        Args:
            batch_size: Number of sequences per batch
            
        Yields:
            Batch objects with input_ids and labels
        """
        batch_sequences = []
        
        for sequence in self:
            batch_sequences.append(sequence)
            
            if len(batch_sequences) >= batch_size:
                yield self._make_batch(batch_sequences)
                batch_sequences = []
        
        # Yield final partial batch if any
        if batch_sequences:
            yield self._make_batch(batch_sequences)
    
    def _make_batch(self, sequences: List[List[int]]) -> Batch:
        """Convert list of sequences to a Batch."""
        # Stack sequences
        arr = np.array(sequences, dtype=np.int32)
        
        # Input is all but last token, labels are all but first
        input_ids = mx.array(arr[:, :-1])
        labels = mx.array(arr[:, 1:])
        
        return Batch(input_ids=input_ids, labels=labels)


class MixedStreamingDataset:
    """
    Mixed streaming dataset that samples from multiple datasets based on weights.
    
    Each dataset is sampled according to its weight, creating a weighted mixture
    of training data.
    """
    
    def __init__(
        self,
        datasets_config: List[Dict[str, Any]],
        split: str = "train",
        seq_len: int = 1024,
        shuffle_buffer: int = 10000,
    ):
        """
        Args:
            datasets_config: List of dicts with keys:
                - name: HuggingFace dataset name
                - config: Optional dataset configuration
                - weight: Sampling weight (will be normalized)
            split: Dataset split
            seq_len: Sequence length
            shuffle_buffer: Shuffle buffer size
        """
        self.datasets_config = datasets_config
        self.split = split
        self.seq_len = seq_len
        self.shuffle_buffer = shuffle_buffer
        self.tokenizer = get_tokenizer()
        self.eos_token = self.tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
        
        # Normalize weights
        total_weight = sum(d.get("weight", 1.0) for d in datasets_config)
        self.weights = [d.get("weight", 1.0) / total_weight for d in datasets_config]
        self.cumulative_weights = []
        cumsum = 0.0
        for w in self.weights:
            cumsum += w
            self.cumulative_weights.append(cumsum)
        
        # Lazy-loaded iterators for each dataset
        self._iterators: List[Optional[Iterator]] = [None] * len(datasets_config)
        self._datasets: List[Optional[Any]] = [None] * len(datasets_config)
        self._buffers: List[List[int]] = [[] for _ in datasets_config]
    
    def _load_dataset(self, idx: int):
        """Lazily load a single dataset."""
        if self._datasets[idx] is not None:
            return
        
        from datasets import DownloadConfig, load_dataset
        
        config = self.datasets_config[idx]
        dataset_name = config["name"]
        dataset_config = config.get("config")
        cache_dir = (Path(__file__).resolve().parent / "cache").expanduser()
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")
        os.environ.setdefault("HF_HUB_HTTP_TIMEOUT", "120")
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
        
        download_config = DownloadConfig(
            cache_dir=str(cache_dir),
            max_retries=5,
        )
        
        if dataset_config:
            dataset = load_dataset(
                dataset_name,
                dataset_config,
                split=self.split,
                streaming=True,
                cache_dir=str(cache_dir),
                download_config=download_config,
            )
        else:
            dataset = load_dataset(
                dataset_name,
                split=self.split,
                streaming=True,
                cache_dir=str(cache_dir),
                download_config=download_config,
            )
        
        if self.shuffle_buffer > 0:
            dataset = dataset.shuffle(buffer_size=self.shuffle_buffer, seed=42 + idx)
        
        self._datasets[idx] = dataset
        self._iterators[idx] = iter(dataset)
    
    def _get_sequence_from_dataset(self, idx: int) -> Optional[List[int]]:
        """Get a complete sequence from dataset at index idx."""
        self._load_dataset(idx)
        
        buffer = self._buffers[idx]
        iterator = self._iterators[idx]
        
        # Fill buffer until we have a complete sequence
        while len(buffer) < self.seq_len + 1:
            try:
                example = next(iterator)
                text = example.get("text", "")
                if text:
                    tokens = self.tokenizer.encode(text)
                    tokens.append(self.eos_token)
                    buffer.extend(tokens)
            except StopIteration:
                # Restart iterator
                self._iterators[idx] = iter(self._datasets[idx])
                iterator = self._iterators[idx]
        
        # Extract sequence
        sequence = buffer[:self.seq_len + 1]
        self._buffers[idx] = buffer[self.seq_len:]
        return sequence
    
    def _select_dataset(self) -> int:
        """Select a dataset index based on weights."""
        r = random.random()
        for i, cw in enumerate(self.cumulative_weights):
            if r <= cw:
                return i
        return len(self.cumulative_weights) - 1
    
    def __iter__(self) -> Iterator[List[int]]:
        """Iterate over packed sequences, sampling from datasets by weight."""
        while True:
            idx = self._select_dataset()
            sequence = self._get_sequence_from_dataset(idx)
            if sequence:
                yield sequence
    
    def get_batch_iterator(self, batch_size: int) -> Iterator[Batch]:
        """Get an iterator that yields batches."""
        batch_sequences = []
        
        for sequence in self:
            batch_sequences.append(sequence)
            
            if len(batch_sequences) >= batch_size:
                yield self._make_batch(batch_sequences)
                batch_sequences = []
    
    def _make_batch(self, sequences: List[List[int]]) -> Batch:
        """Convert list of sequences to a Batch."""
        arr = np.array(sequences, dtype=np.int32)
        input_ids = mx.array(arr[:, :-1])
        labels = mx.array(arr[:, 1:])
        return Batch(input_ids=input_ids, labels=labels)


class ValidationDataset:
    """
    Fixed validation dataset (not streaming).
    
    Loads a subset of data for consistent validation.
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        seq_len: int = 1024,
        dataset_name: str = "wikitext",
        dataset_config: Optional[str] = "wikitext-103-raw-v1",
    ):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.tokenizer = get_tokenizer()
        self.eos_token = self.tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
        
        self._sequences = None
    
    def _load(self):
        """Load validation sequences."""
        if self._sequences is not None:
            return
        
        from datasets import load_dataset
        
        # Try validation split first, fall back to train split if not available
        try:
            if self.dataset_config:
                dataset = load_dataset(
                    self.dataset_name,
                    self.dataset_config,
                    split="validation",
                    streaming=True,
                )
            else:
                dataset = load_dataset(
                    self.dataset_name,
                    split="validation",
                    streaming=True,
                )
        except ValueError:
            # No validation split, use a portion of train split
            if self.dataset_config:
                dataset = load_dataset(
                    self.dataset_name,
                    self.dataset_config,
                    split="train",
                    streaming=True,
                )
            else:
                dataset = load_dataset(
                    self.dataset_name,
                    split="train",
                    streaming=True,
                )
            # Skip some samples to avoid overlap with training data
            dataset = dataset.skip(100000)
        
        sequences = []
        buffer = []
        
        for example in dataset:
            text = example.get("text", "")
            if not text:
                continue
            
            tokens = self.tokenizer.encode(text)
            tokens.append(self.eos_token)
            buffer.extend(tokens)
            
            while len(buffer) >= self.seq_len + 1:
                sequence = buffer[:self.seq_len + 1]
                buffer = buffer[self.seq_len:]
                sequences.append(sequence)
                
                if len(sequences) >= self.num_samples:
                    break
            
            if len(sequences) >= self.num_samples:
                break
        
        self._sequences = sequences
    
    def get_batches(self, batch_size: int) -> List[Batch]:
        """Get all validation batches."""
        self._load()
        
        batches = []
        for i in range(0, len(self._sequences), batch_size):
            batch_seqs = self._sequences[i:i + batch_size]
            arr = np.array(batch_seqs, dtype=np.int32)
            input_ids = mx.array(arr[:, :-1])
            labels = mx.array(arr[:, 1:])
            batches.append(Batch(input_ids=input_ids, labels=labels))
        
        return batches


def create_dataloader(
    split: str = "train",
    batch_size: int = 4,
    seq_len: int = 1024,
    shuffle_buffer: int = 10000,
    dataset_name: str = "wikitext",
    dataset_config: Optional[str] = "wikitext-103-raw-v1",
    datasets_config: Optional[List[Dict[str, Any]]] = None,
    streaming: bool = True,
) -> Iterator[Batch]:
    """
    Create a data loader.
    
    Args:
        split: Dataset split ("train" or "validation")
        batch_size: Batch size
        seq_len: Sequence length
        shuffle_buffer: Size of shuffle buffer (0 to disable)
        dataset_name: HuggingFace dataset name or path (fallback if datasets_config not provided)
        dataset_config: Optional dataset configuration (e.g., subset)
        datasets_config: Optional list of dataset configs for mixed training.
            Each dict should have: name, config (optional), weight
        streaming: Whether to use streaming (ignored, always streams)
        
    Returns:
        Iterator yielding Batch objects
    """
    if split == "validation":
        dataset = ValidationDataset(
            num_samples=1000,
            seq_len=seq_len,
            dataset_name=dataset_name,
            dataset_config=dataset_config,
        )
        return iter(dataset.get_batches(batch_size))
    
    # Use mixed dataset if datasets_config is provided
    if datasets_config:
        dataset = MixedStreamingDataset(
            datasets_config=datasets_config,
            split=split,
            seq_len=seq_len,
            shuffle_buffer=shuffle_buffer,
        )
        return dataset.get_batch_iterator(batch_size)
    
    # Fallback to single dataset
    dataset = StreamingTextDataset(
        split=split,
        seq_len=seq_len,
        shuffle_buffer=shuffle_buffer,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
    )
    return dataset.get_batch_iterator(batch_size)


def estimate_tokens_per_epoch() -> int:
    """
    Estimate total tokens in the default training set.
    
    Based on WikiText-103 statistics: ~100M tokens
    """
    return 100_000_000

