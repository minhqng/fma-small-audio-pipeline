"""
Quick verification script for the data ingestion pipeline.

Usage:  python -m src.ingestion.verify
"""
from __future__ import annotations

import traceback
from pathlib import Path

from src.ingestion.metadata import build_metadata, save_label_map, validate_audio_files
from src.utils.helpers import load_config

def main() -> None:
    cfg = load_config()

    print("--- Starting data verification ---")

    try:
        # 1. Build metadata (config-driven)
        df, lm = build_metadata(cfg)
        save_label_map(lm, Path(cfg.data.label_map_path))

        print("Metadata loaded successfully!")
        print(f"   - Genres: {len(lm)}")
        print(f"   - Genre list: {lm}")
        print(f"   - Initial tracks: {df.shape[0]}")

        # 2. Validate audio files (fail-fast + silence check)
        print("\n--- Validating audio files ---")
        df_valid = validate_audio_files(
            df,
            target_sr=cfg.data.sample_rate,
            min_duration_s=cfg.audio.min_duration_s,
            silence_threshold_db=cfg.audio.silence_threshold_db,
        )

        print("Validation results:")
        print(f"   - Valid tracks: {df_valid.shape[0]}")
        print(f"   - Unique artists: {df_valid['artist_id'].n_unique()}")

        # 3. Assertions
        assert len(lm) == cfg.data.num_classes, (
            f"Expected {cfg.data.num_classes} genres but got {len(lm)}"
        )
        assert df_valid.shape[0] > 0, "No valid tracks found!"

        print("\nALL CHECKS PASSED!")

    except Exception as e:
        print(f"\nERROR: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
