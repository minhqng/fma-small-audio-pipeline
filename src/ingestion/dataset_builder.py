"""
Dataset builder: Arrow dataset creation, HF Hub push,
and Stratified Group K-Fold splitting.
"""
from __future__ import annotations

from collections.abc import Generator
from typing import Any

import datasets
import numpy as np
import polars as pl
import torch
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedGroupKFold

from src.features.audio_transforms import AudioTransform

#
# Segment-Level Silence Gate
#

_SILENCE_RMS_FLOOR = 1e-7  # absolute RMS below which a mel segment is silent

def _is_silent_segment(mel: torch.Tensor) -> bool:
    """
    Reject mel-spectrogram segments that are effectively silent.
    Operates on the linear-scale mel (before log transform),
    but here we receive post-log mel. We check if variance ~ 0,
    meaning the segment is constant (silent produces flat log-mel).
    """
    return bool(mel.std().item() < _SILENCE_RMS_FLOOR)

def _sample_generator(
    df: pl.DataFrame,
    transform: AudioTransform,
    label_map: dict[str, int],
) -> Generator[dict[str, Any], None, None]:
    """
    Generator that yields one dict per mel-spectrogram segment.
    HF Datasets calls this to build the Arrow table row-by-row,
    avoiding loading all audio into RAM at once.

    Includes per-segment silence gate: silent segments are skipped.
    """
    for row in df.iter_rows(named=True):
        try:
            mels = transform.process_track(row["file_path"])
        except Exception as e:
            print(f"[WARN] Skipping track {row['track_id']}: {e}")
            continue

        for mel in mels:
            # Fail-fast: reject silent segments
            if _is_silent_segment(mel):
                continue

            yield {
                "mel": mel.squeeze(0).cpu().numpy(),  # (128, 300) float32
                "label": row["label"],
                "track_id": row["track_id"],
                "artist_id": row["artist_id"],
                "genre": row["genre_top"],
            }

#
# Arrow Dataset Construction
#

def build_arrow_dataset(
    df: pl.DataFrame,
    cfg: DictConfig,
    device: torch.device,
    label_map: dict[str, int],
) -> datasets.Dataset:
    """
    Build a Hugging Face Dataset backed by Arrow from the Polars metadata.
    Uses generator mode to avoid loading all audio into RAM.
    """
    transform = AudioTransform.from_config(cfg, device)

    genre_names = sorted(label_map.keys())
    features = datasets.Features({
        "mel": datasets.Array2D(
            shape=(cfg.audio.n_mels, cfg.audio.target_frames),
            dtype="float32",
        ),
        "label": datasets.ClassLabel(names=genre_names),
        "track_id": datasets.Value("int64"),
        "artist_id": datasets.Value("int64"),
        "genre": datasets.Value("string"),
    })

    ds = datasets.Dataset.from_generator(
        _sample_generator,
        features=features,
        gen_kwargs={
            "df": df,
            "transform": transform,
            "label_map": label_map,
        },
        cache_dir=cfg.data.processed_path,
    )

    print(f"[INFO] Dataset built: {len(ds)} samples")
    return ds

#
# Hub Push
#

def push_to_hub(
    ds: datasets.Dataset | datasets.DatasetDict,
    repo_id: str,
    token: str,
    private: bool = True,
) -> None:
    """Push dataset to Hugging Face Hub as a private repo."""
    ds.push_to_hub(
        repo_id=repo_id,
        token=token,
        private=private,
    )
    print(f"[INFO] Dataset pushed to hub: {repo_id}")

#
# Stratified Group K-Fold Split (Zero-Leakage)
#

def split_dataset(
    repo_id: str,
    token: str,
    seed: int,
) -> datasets.DatasetDict:
    """
    Load full dataset from Hub, apply Stratified Group K-Fold split
    on artist_id to prevent data leakage, push DatasetDict back.

    Strategy:
      1. 5-fold StratifiedGroupKFold -> fold 0: 80% trainval / 20% test
      2. On trainval: 5-fold again  -> fold 0: 80% train / 20% val
      => Final approximate ratio: ~64% train, ~16% val, ~20% test

    All segments from the same artist_id land in the SAME split.
    """
    # Load full dataset from Hub (streaming=False needed for split indices)
    ds = datasets.load_dataset(repo_id, token=token, split="train")

    labels = np.array(ds["label"])
    groups = np.array(ds["artist_id"])

    # Split 1: trainval vs test
    sgkf = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=seed,
    )
    trainval_idx, test_idx = next(sgkf.split(
        X=np.arange(len(ds)), y=labels, groups=groups,
    ))

    ds_trainval = ds.select(trainval_idx.tolist())
    ds_test = ds.select(test_idx.tolist())

    # Split 2: train vs val (from trainval)
    labels_tv = np.array(ds_trainval["label"])
    groups_tv = np.array(ds_trainval["artist_id"])

    sgkf2 = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=seed,
    )
    train_idx, val_idx = next(sgkf2.split(
        X=np.arange(len(ds_trainval)), y=labels_tv, groups=groups_tv,
    ))

    ds_train = ds_trainval.select(train_idx.tolist())
    ds_val = ds_trainval.select(val_idx.tolist())

    dd = datasets.DatasetDict({
        "train": ds_train,
        "validation": ds_val,
        "test": ds_test,
    })

    #  Zero-Leakage Assertions
    train_artists = set(ds_train["artist_id"])
    val_artists = set(ds_val["artist_id"])
    test_artists = set(ds_test["artist_id"])
    assert train_artists.isdisjoint(test_artists), (
        "LEAKAGE: train and test share artists!"
    )
    assert train_artists.isdisjoint(val_artists), (
        "LEAKAGE: train and val share artists!"
    )
    assert val_artists.isdisjoint(test_artists), (
        "LEAKAGE: val and test share artists!"
    )

    print(
        f"[INFO] Split sizes -- Train: {len(ds_train)}, "
        f"Val: {len(ds_val)}, Test: {len(ds_test)}"
    )
    print("[INFO] Artist leakage check PASSED")

    # Push back as DatasetDict with splits
    push_to_hub(dd, repo_id=repo_id, token=token, private=True)

    return dd
