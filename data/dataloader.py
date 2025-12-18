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
from typing import Iterator, Optional, List, Tuple
from dataclasses import dataclass
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
) -> Iterator[Batch]:
    """
    Create a data loader.
    
    Args:
        split: Dataset split ("train" or "validation")
        batch_size: Batch size
        seq_len: Sequence length
        shuffle_buffer: Size of shuffle buffer (0 to disable)
        dataset_name: HuggingFace dataset name or path
        dataset_config: Optional dataset configuration (e.g., subset)
        
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
    else:
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

