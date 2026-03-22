"""
Data Engineering Pipeline Orchestrator

Run:  python main.py
"""
from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from src.ingestion.dataset_builder import (
    build_arrow_dataset,
    push_to_hub,
    split_dataset,
)
from src.ingestion.metadata import (
    build_metadata,
    save_label_map,
    validate_audio_files,
)
from src.utils.helpers import (
    compute_global_stats,
    get_device,
    get_hf_token,
    load_config,
    seed_everything,
)


def main() -> None:
    # Phase 0 -- Setup (all constants from Hydra config)
    cfg = load_config("configs/config.yaml")
    seed_everything(cfg.training.seed)
    device = get_device()
    token = get_hf_token(cfg)
    print(OmegaConf.to_yaml(cfg))

    # Phase 1 -- Ingestion: Parse metadata (Polars Sandwich Strategy)
    print("\n=== PHASE 1: Data Ingestion ===")
    df, label_map = build_metadata(cfg)
    save_label_map(label_map, Path(cfg.data.label_map_path))

    # Phase 2 -- Validation: Fail-fast corrupt + silence detection
    print("\n=== PHASE 2: Audio Validation ===")
    df = validate_audio_files(
        df,
        target_sr=cfg.data.sample_rate,
        min_duration_s=cfg.audio.min_duration_s,
        silence_threshold_db=cfg.audio.silence_threshold_db,
    )

    # Phase 3 -- Dataset Build: Arrow + Push to Hub
    print("\n=== PHASE 3: Dataset Build & Push ===")
    ds = build_arrow_dataset(df, cfg, device, label_map)
    push_to_hub(
        ds,
        repo_id=cfg.hub.repo_id,
        token=token,
        private=cfg.hub.private,
    )

    # Phase 4 -- Splitting: Stratified Group K-Fold (Zero-Leakage)
    print("\n=== PHASE 4: Stratified Group Split ===")
    dd = split_dataset(
        repo_id=cfg.hub.repo_id,
        token=token,
        seed=cfg.training.seed,
    )

    # Phase 5 -- Handover: Compute global stats (streaming, Welford)
    print("\n=== PHASE 5: Global Stats ===")
    stats = compute_global_stats(cfg=cfg, token=token)

    # Summary
    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print(f"  Dataset:    {cfg.hub.repo_id}")
    print(f"  Splits:     {list(dd.keys())}")
    print(f"  Train:      {len(dd['train'])} samples")
    print(f"  Val:        {len(dd['validation'])} samples")
    print(f"  Test:       {len(dd['test'])} samples")
    print(f"  Stats:      {len(stats['mean'])} mel bins")
    print(f"  Label Map:  {label_map}")
    print("=" * 60)

if __name__ == "__main__":
    main()
