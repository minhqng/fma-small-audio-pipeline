<div align="center">

# fma-small-audio-pipeline

A code-first, reproducible Python pipeline for turning raw FMA Small audio into validated, fixed-size log-mel spectrogram segments and Hugging Face `datasets` artifacts for music genre classification.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](./pyproject.toml)
[![CI](https://github.com/minhqng/fma-small-audio-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/minhqng/fma-small-audio-pipeline/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](./LICENSE)

</div>

## What This Project Is

`fma-small-audio-pipeline` is a reproducible data pipeline for the [FMA Small](https://github.com/mdeff/fma) dataset. It is built around Python modules and config files, not notebooks: the notebook in this repo is a companion demo, while the main workflow lives in [main.py](./main.py) and [src/](./src).

The pipeline:

- parses the multi-header `tracks.csv` metadata
- validates raw audio for existence, duration, and silence
- extracts `(128, 300)` log-mel spectrogram segments from 3-second windows
- builds an Arrow-backed Hugging Face `datasets` dataset
- applies artist-aware `train` / `validation` / `test` splits
- computes global normalization statistics for downstream training

## Highlights

- Reproducible, config-driven workflow centered on [configs/config.yaml](./configs/config.yaml)
- **Fail-fast audio validation** for missing, corrupted, short, or silent files before dataset build
- **Fixed-shape features** exported as `(128, 300)` log-mel spectrograms for downstream modeling
- **Segment-level filtering** to drop unusable silent segments during dataset construction
- **Artist-aware splitting** to reduce train/validation/test leakage across the same creator
- **Hugging Face-ready output** in Arrow format for efficient loading and reuse
- **Training-only normalization stats** computed in a streaming fashion from the train split
- Companion resources: a published dataset on Hugging Face and a demo notebook for inspecting the processed artifacts

## Pipeline Overview

The full pipeline runs in six stages:

1. Parse FMA metadata and build the genre label map.
2. Validate raw audio files for existence, corruption, minimum duration, and silence.
3. Extract fixed-size log-mel spectrogram segments from each valid track.
4. Filter unusable silent segments and export the processed samples as a Hugging Face Arrow dataset.
5. Create artist-aware train/validation/test splits with a two-stage group-aware split strategy.
6. Compute streaming mean and standard deviation statistics from the training split only.

## Quick Start

### 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### 2. Set Up the Environment

Copy [.env.example](./.env.example) to `.env` and set a Hugging Face token:

```bash
HF_TOKEN=hf_your_token_here
```

`HF_TOKEN` is required for `python main.py`, because the full pipeline pushes to the Hugging Face Hub and then reloads the dataset to create splits and compute statistics. The validation-only command below does not require a token.

Before running the full pipeline, review [configs/config.yaml](./configs/config.yaml) and set `hub.repo_id` to a dataset repo you can write to. The checked-in default points to `minhqng/fma-small`.

### 3. Place the FMA Data

The repository does not redistribute raw FMA audio files. Obtain the FMA Small metadata and audio separately and place them in the default layout:

```text
data/
`-- raw/
    |-- fma_metadata/
    |   `-- tracks.csv
    `-- fma_small/
        |-- 000/000002.mp3
        |-- 000/000005.mp3
        `-- ...
```

### 4. Run Validation Only

```bash
python -m src.ingestion.verify
```

This loads metadata, writes [data/label_map.json](./data/label_map.json), and validates the raw audio files before any feature extraction.

### 5. Run the Full Pipeline

```bash
python main.py
```

This builds the Arrow-backed dataset, pushes it to the configured Hub repo, applies artist-aware `train` / `validation` / `test` splits, and writes global normalization statistics.

## Outputs

| Artifact | Default path | Purpose |
| --- | --- | --- |
| Label map | [data/label_map.json](./data/label_map.json) | Genre-to-integer mapping used during dataset creation |
| Arrow dataset build cache | `data/processed/fma_arrow/` | Arrow-backed `datasets` build/cache directory used during dataset construction |
| Normalization stats | [data/stats.json](./data/stats.json) | Global per-mel-bin mean and standard deviation computed from the training split |
| Published dataset | [minhqng/fma-small](https://huggingface.co/datasets/minhqng/fma-small) | Processed dataset that can be inspected or loaded from Hugging Face |

## Inspect the Published Dataset

The published dataset reflects the pipeline outputs after feature extraction, segment filtering, artist-aware splitting, and export to Arrow format.
An already-built version of the processed dataset is available on Hugging Face: [minhqng/fma-small](https://huggingface.co/datasets/minhqng/fma-small).

```python
from datasets import load_dataset

dd = load_dataset("minhqng/fma-small")
```

If you want to reproduce the pipeline yourself, keep the published dataset for inspection and point `hub.repo_id` at a repo you control for your own runs.

## Demo Notebook

The repository includes a companion notebook, [notebooks/fma_pipeline_demo.ipynb](./notebooks/fma_pipeline_demo.ipynb), that shows how to inspect the pipeline stages and work with the processed dataset artifacts. It is a demo surface, not the primary interface.

## Resources

- Published dataset: [Hugging Face dataset page](https://huggingface.co/datasets/minhqng/fma-small)
- Demo notebook: [notebooks/fma_pipeline_demo.ipynb](https://colab.research.google.com/drive/1iVXZAuHvL52auQ-pNElF63S6eiS1YUht?usp=sharing)
- CI workflow: [.github/workflows/ci.yml](./.github/workflows/ci.yml)

## Configuration

Runtime settings live in [configs/config.yaml](./configs/config.yaml). The most important groups are:

- `data`: raw input locations, processed output locations, subset selection, and artifact paths
- `audio`: sample rate, mel settings, segment duration, overlap, frame target, and validation thresholds
- `hub`: Hugging Face dataset repo, privacy setting, and token environment variable
- `training`: reproducibility seed and downstream training defaults

Out of the box, the config targets:

- Python `3.11+`
- FMA metadata at `data/raw/fma_metadata/tracks.csv`
- FMA audio at `data/raw/fma_small/`
- Arrow build/cache output under `data/processed/fma_arrow/`
- label map at `data/label_map.json`
- normalization statistics at `data/stats.json`

## Project Layout

```text
.
|-- configs/
|   `-- config.yaml
|-- data/
|   |-- raw/
|   |   |-- fma_metadata/
|   |   `-- fma_small/
|   `-- processed/
|-- notebooks/
|   `-- fma_pipeline_demo.ipynb
|-- src/
|   |-- features/
|   |-- ingestion/
|   `-- utils/
|-- tests/
|-- main.py
`-- pyproject.toml
```

## Development

Local checks:

```bash
python -m ruff check main.py src tests
python -m mypy src tests
python -m pytest
```

Continuous integration is defined in [.github/workflows/ci.yml](./.github/workflows/ci.yml) and runs the same checks on pushes and pull requests.

## Notes on Data and Licensing

- This repository contains pipeline code and lightweight scaffolding only.
- Raw FMA audio files are not redistributed here; you must obtain the FMA data separately.
- The code license for this repository and the upstream dataset licensing are separate concerns.
- Before sharing or publishing derived data, review the licensing terms that apply to the original FMA dataset and any downstream artifacts you create.

## License

This repository's code is licensed under the Apache License 2.0. See [LICENSE](./LICENSE).
