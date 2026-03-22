from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import av
import numpy as np
import polars as pl
from omegaconf import DictConfig
from tqdm import tqdm

#
# Phase 1  Polars "Sandwich Strategy" for FMA multi-header CSV
#

def _build_flattened_headers(csv_path: Path) -> list[str]:
    """
    Read the first 3 rows of FMA tracks.csv to construct
    flattened column names in the format 'category.attribute'.

    FMA tracks.csv structure:
      Row 0: top-level categories (e.g., '', 'album', 'artist', 'track', ...)
      Row 1: sub-level attributes (e.g., '', 'comments', 'id', 'genre_top', ...)
      Row 2: sub-sub-level (mostly empty, ignored)

    Returns a list like: ['track_id', 'album.comments', 'album.date_created', ...,
                          'artist.id', ..., 'track.genre_top', ..., 'set.subset', ...]
    """
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        row0 = next(reader)  # categories
        row1 = next(reader)  # attributes
        # row2 is sub-sub (mostly empty), skip it

    # Forward-fill row0: FMA leaves category blank if same as previous
    filled_categories: list[str] = []
    current_cat = ""
    for raw_cat in row0:
        stripped = raw_cat.strip()
        if stripped:
            current_cat = stripped
        filled_categories.append(current_cat)

    # Build flattened names
    headers: list[str] = []
    for i, (raw_c, raw_a) in enumerate(zip(filled_categories, row1)):
        cat = raw_c.strip()
        attr = raw_a.strip()
        if i == 0:
            # First column is always the unnamed index = track_id
            headers.append("track_id")
        elif cat and attr:
            headers.append(f"{cat}.{attr}")
        elif attr:
            headers.append(attr)
        else:
            headers.append(f"_unnamed_{i}")

    return headers

def parse_fma_tracks(csv_path: Path, subset: str = "small") -> pl.DataFrame:
    """
    Parse FMA tracks.csv which has a 3-row multi-level header.

    Strategy (Polars Sandwich):
      1. Use stdlib csv to read 3 header rows -> build flattened column names
      2. Use Polars to read data body (skip 3 rows), assign flattened names
      3. Select needed columns by NAME (not index)
      4. Filter to requested subset FIRST, then drop nulls

    Required columns:
      - track_id        (row index, first column)
      - track.genre_top (the top-level genre label)
      - artist.id       (for stratified group splitting)
      - set.subset      (to filter 'small' / 'medium' / 'large')
    """
    # Step 1: Build flattened header names from the 3 header rows
    flat_headers = _build_flattened_headers(csv_path)

    # Step 2: Read data body with Polars, skipping the 3 header rows
    df = pl.read_csv(
        csv_path,
        skip_rows=3,
        has_header=False,
        infer_schema_length=0,  # read all as String first
        null_values=[""],
    )

    # Rename to flattened headers
    current_cols = df.columns
    if len(current_cols) != len(flat_headers):
        raise ValueError(
            f"Column count mismatch: CSV body has {len(current_cols)} cols "
            f"but header parsing found {len(flat_headers)} names. "
            f"Check tracks.csv format."
        )

    rename_map = dict(zip(current_cols, flat_headers))
    df = df.rename(rename_map)

    # Step 3: Find target columns by name matching
    print(f"[DEBUG] Total columns: {len(df.columns)}")

    target_cols: dict[str, str | None] = {
        "track_id": "track_id",
        "genre_top": None,
        "artist_id": None,
        "subset": None,
    }

    for col_name in df.columns:
        col_lower = col_name.lower()
        if "genre_top" in col_lower:
            target_cols["genre_top"] = col_name
        if col_lower == "artist.id":
            target_cols["artist_id"] = col_name
        if col_lower == "set.subset":
            target_cols["subset"] = col_name

    # Validate all targets found
    for key, val in target_cols.items():
        if val is None:
            print(f"[ERROR] Could not find '{key}'. Available columns:")
            for i, c in enumerate(df.columns):
                sample = df[c].head(1).to_list()[0]
                print(f"  [{i:3d}] {c} = {sample}")
            raise KeyError(
                f"Required column '{key}' not found in flattened headers"
            )

    print(f"[INFO] Mapped columns: {target_cols}")

    # Step 4: Select and cast
    df = df.select([
        pl.col(target_cols["track_id"]).cast(pl.Int64).alias("track_id"),
        pl.col(target_cols["genre_top"]).str.strip_chars().alias("genre_top"),
        pl.col(target_cols["artist_id"]).cast(pl.Int64).alias("artist_id"),
        pl.col(target_cols["subset"]).str.strip_chars().alias("subset"),
    ])

    #
    # CRITICAL: Filter to requested subset BEFORE building label map
    # FMA tracks.csv has all subsets: small (8k), medium (25k), large (106k)
    # If we don't filter here, the label map gets 16 genres instead of 8
    #
    pre_filter_count = df.shape[0]
    df = df.filter(pl.col("subset") == subset)
    print(
        f"[INFO] Subset filter '{subset}': "
        f"{pre_filter_count} -> {df.shape[0]} tracks"
    )

    # Drop the subset column (no longer needed) and null genres
    df = df.drop("subset").filter(
        pl.col("genre_top").is_not_null()
        & (pl.col("genre_top") != "")
    )

    return df

def build_label_map(df: pl.DataFrame) -> dict[str, int]:
    """Create sorted genre -> integer label mapping."""
    genres = sorted(df["genre_top"].unique().to_list())
    return {genre: idx for idx, genre in enumerate(genres)}

def resolve_audio_path(audio_path: Path, track_id: int) -> str:
    """
    FMA directory convention: track 012345 -> 012/012345.mp3
    Track IDs are zero-padded to 6 digits, folder is first 3 digits.
    """
    tid_str = f"{track_id:06d}"
    return str(audio_path / tid_str[:3] / f"{tid_str}.mp3")

def save_label_map(label_map: dict[str, int], path: Path) -> None:
    """Persist label map to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Label map saved to {path}")

#
# Phase 2  Fail-Fast Audio Validation (PyAV + Silence Check)
#

def _compute_rms_db(file_path: str) -> float:
    """
    Compute RMS energy in dB for an audio file using PyAV.

    Returns RMS in dBFS (0 dB = full scale). Silent files return -inf.
    """
    container = av.open(file_path)
    frames: list[np.ndarray] = []
    for frame in container.decode(audio=0):
        frames.append(frame.to_ndarray())
    container.close()

    if not frames:
        return -math.inf

    audio = np.concatenate(frames, axis=1).astype(np.float32)
    rms = float(np.sqrt(np.mean(audio ** 2)))
    rms_floor = 1e-10
    if rms < rms_floor:
        return -math.inf
    return 20.0 * math.log10(rms)

def validate_audio_files(
    df: pl.DataFrame,
    target_sr: int,
    min_duration_s: float = 1.0,
    silence_threshold_db: float = -60.0,
) -> pl.DataFrame:
    """
    Fail-fast validation: use PyAV to probe audio files.

    Three-layer defense:
      1. Physical existence (already handled by build_metadata)
      2. Duration check: must be >= min_duration_s
      3. Silence check: RMS energy must be > silence_threshold_db

    Args:
        df: DataFrame with 'file_path' and 'track_id' columns.
        target_sr: Expected sample rate (logged for diagnostics).
        min_duration_s: Minimum valid audio duration in seconds.
        silence_threshold_db: RMS threshold in dB; files below are rejected.

    Returns:
        Filtered DataFrame containing only valid, audible files.
    """
    valid_ids: list[int] = []
    corrupt_count = 0
    silent_count = 0
    sr_mismatch_count = 0

    for row in tqdm(
        df.iter_rows(named=True),
        total=df.shape[0],
        desc="Validating audio",
    ):
        try:
            container = av.open(row["file_path"])
            stream = container.streams.audio[0]
            duration_s = float(stream.duration * stream.time_base)
            file_sr = stream.rate
            container.close()

            # Check 1: Duration
            if duration_s < min_duration_s:
                corrupt_count += 1
                continue

            # Check 2: Sample rate diagnostic (warn, don't reject
            # PyAV resamples on decode, torchaudio.Resample handles the rest)
            if file_sr != target_sr:
                sr_mismatch_count += 1

            # Check 3: Silence detection via RMS
            rms_db = _compute_rms_db(row["file_path"])
            if rms_db < silence_threshold_db:
                silent_count += 1
                continue

            valid_ids.append(row["track_id"])
        except Exception:
            corrupt_count += 1

    print(
        f"[INFO] Validation done: {len(valid_ids)} valid, "
        f"{corrupt_count} corrupt/short, {silent_count} silent, "
        f"{sr_mismatch_count} SR mismatches (non-blocking)"
    )
    return df.filter(pl.col("track_id").is_in(valid_ids))

#
# Orchestrator  Config-Driven Build
#

def build_metadata(cfg: DictConfig) -> tuple[pl.DataFrame, dict[str, int]]:
    """
    Full ingestion orchestrator (config-driven).

    Steps:
      1. Parse tracks.csv from metadata_path (filtered to configured subset)
      2. Build label map (genre -> int)
      3. Add file_path column using audio_path
      4. Validate file existence (fail-fast on missing files)
      5. Add integer label column

    Args:
        cfg: The full Hydra DictConfig (needs cfg.data.* fields).

    Returns:
        (df, label_map) tuple.
    """
    metadata_path = Path(cfg.data.metadata_path)
    audio_path = Path(cfg.data.audio_path)
    subset = cfg.data.subset

    csv_path = metadata_path / "tracks.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"tracks.csv not found at {csv_path}")

    # Step 1: Parse (already filtered to configured subset)
    df = parse_fma_tracks(csv_path, subset=subset)

    # Step 2: Label map
    label_map = build_label_map(df)

    # Validate expected genre count from config
    if len(label_map) != cfg.data.num_classes:
        print(
            f"[WARN] Expected {cfg.data.num_classes} genres for "
            f"fma_{subset}, got {len(label_map)}: {list(label_map.keys())}"
        )

    # Step 3: Add file paths and labels
    df = df.with_columns([
        pl.col("track_id").map_elements(
            lambda tid: resolve_audio_path(audio_path, tid),
            return_dtype=pl.Utf8,
        ).alias("file_path"),
        pl.col("genre_top").replace(label_map).cast(pl.Int64).alias("label"),
    ])

    # Step 4: Fail-fast file existence validation
    file_paths = df["file_path"].to_list()
    missing = [fp for fp in file_paths if not Path(fp).exists()]
    if missing:
        print(
            f"[WARN] {len(missing)} / {len(file_paths)} "
            f"audio files missing. Filtering out."
        )
        existing_set = set(file_paths) - set(missing)
        df = df.filter(
            pl.col("file_path").is_in(list(existing_set))
        )

    print(
        f"[INFO] Metadata ready: {df.shape[0]} tracks, "
        f"{len(label_map)} genres"
    )
    return df, label_map
