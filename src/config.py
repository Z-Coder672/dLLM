"""
Model and training configuration dataclasses.
"""

from dataclasses import dataclass, field
from typing import Optional
import yaml


@dataclass
class ModelConfig:
    """Configuration for the Ternary Transformer model."""
    
    # Architecture
    d_model: int = 1024
    n_layers: int = 24
    n_heads: int = 16
    d_ff: int = 4096
    vocab_size: int = 50257  # GPT-2 tokenizer
    max_seq_len: int = 1024
    
    # Quantization
    threshold_factor: float = 0.7  # τ = factor * mean(|W|)
    ternary_temperature: float = 0.15
    
    # Dropout (typically 0 for small models)
    dropout: float = 0.0
    
    # RoPE settings
    rope_theta: float = 10000.0
    
    # Precision
    dtype: str = "bfloat16"
    
    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads
    
    @classmethod
    def from_yaml(cls, path: str) -> "ModelConfig":
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict.get("model", {}))
    
    def to_yaml(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.dump({"model": self.__dict__}, f, default_flow_style=False)


@dataclass 
class TrainingConfig:
    """Configuration for training."""
    
    # Optimization
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-5
    warmup_steps: int = 5000
    weight_decay: float = 0.1
    gradient_clip: float = 0.5
    ternary_skip_steps: int = 1000
    ternary_fadein_steps: int = 0
    
    # Batch settings
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    
    # Training duration
    max_steps: int = 100000
    eval_interval: int = 500
    save_interval: int = 1000
    log_interval: int = 10
    
    # INT8 quantization
    scale_update_interval: int = 500
    
    # Gradient checkpointing
    checkpoint_layers: int = 6  # Checkpoint every N layers
    
    # Data
    dataset_name: str = "wikitext"
    dataset_config: Optional[str] = "wikitext-103-raw-v1"
    sequence_length: int = 1024
    
    # Paths
    output_dir: str = "checkpoints"
    
    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps
    
    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict.get("training", {}))
    
    def to_yaml(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.dump({"training": self.__dict__}, f, default_flow_style=False)


# Preset configurations
CONFIG_500M = ModelConfig(
    d_model=1024,
    n_layers=24,
    n_heads=16,
    d_ff=4096,
    vocab_size=50257,
    max_seq_len=1024,
)

CONFIG_1B = ModelConfig(
    d_model=2048,
    n_layers=24,
    n_heads=16,
    d_ff=8192,
    vocab_size=50257,
    max_seq_len=2048,
)

