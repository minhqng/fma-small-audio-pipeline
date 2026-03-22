from __future__ import annotations

import json
import os
import random
from pathlib import Path

import datasets
import numpy as np
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

def seed_everything(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    """Return best available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config(config_path: str = "configs/config.yaml") -> DictConfig:
    """Load Hydra/OmegaConf config from YAML."""
    return OmegaConf.load(config_path)

def get_hf_token(cfg: DictConfig | None = None) -> str:
    """
    Retrieve Hugging Face token from environment.

    Reads the env variable name from cfg.hub.token_env if provided,
    otherwise defaults to HF_TOKEN.
    """
    load_dotenv()
    env_var = "HF_TOKEN"
    if cfg is not None:
        env_var = getattr(cfg.hub, "token_env", "HF_TOKEN")
    token = os.environ.get(env_var, "")
    if not token:
        raise OSError(
            f"{env_var} not set. Export it or add to .env file."
        )
    return token

def compute_global_stats(cfg: DictConfig, token: str) -> dict[str, list[float]]:
    """
    Compute per-frequency-bin mean and std over the TRAINING split
    using streaming mode (IterableDataset) and Welford's online algorithm.

    This avoids loading the full dataset into RAM -- pure one-pass streaming.

    Args:
        cfg: Full Hydra config (needs cfg.hub.repo_id, cfg.audio.n_mels,
             cfg.data.stats_path).
        token: HF Hub token.

    Returns and saves: {"mean": [n_mels floats], "std": [n_mels floats],
                        "num_samples": int}
    """
    repo_id = cfg.hub.repo_id
    n_mels = cfg.audio.n_mels
    stats_output_path = cfg.data.stats_path

    ds = datasets.load_dataset(
        repo_id, token=token, split="train", streaming=True,
    )

    count = 0
    mean = np.zeros(n_mels, dtype=np.float64)
    m2 = np.zeros(n_mels, dtype=np.float64)

    for sample in tqdm(ds, desc="Computing global stats (streaming)"):
        mel = np.array(sample["mel"], dtype=np.float64)  # (n_mels, frames)
        # Per-frequency-bin mean across time axis
        sample_mean = mel.mean(axis=1)  # (n_mels,)

        count += 1
        delta = sample_mean - mean
        mean += delta / count
        delta2 = sample_mean - mean
        m2 += delta * delta2

    min_samples = 2
    if count < min_samples:
        raise ValueError("Not enough samples to compute statistics")

    variance = m2 / (count - 1)
    std = np.sqrt(variance)

    stats: dict[str, list[float] | int] = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "num_samples": count,
    }

    Path(stats_output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(stats_output_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(
        f"[INFO] Global stats computed over {count} samples, "
        f"saved to {stats_output_path}"
    )
    return stats
