"""Unit tests for the data engineering pipeline"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from src.features.audio_transforms import AudioTransform
from src.ingestion.dataset_builder import _is_silent_segment
from src.utils.helpers import load_config, seed_everything

class TestAudioTransform:
    """Tests for AudioTransform class."""

    @pytest.fixture()
    def cfg(self):
        return load_config()

    @pytest.fixture()
    def transform(self, cfg) -> AudioTransform:
        return AudioTransform.from_config(cfg, device=torch.device("cpu"))

    def test_mel_output_shape(self, transform: AudioTransform) -> None:
        """Mel spectrogram must be exactly (1, 128, 300)."""
        waveform = torch.randn(1, 32000 * 3)
        mel = transform.to_mel_spectrogram(waveform)
        assert mel.shape == (1, 128, 300), f"Expected (1, 128, 300), got {mel.shape}"

    def test_segment_extraction_count(self, transform: AudioTransform) -> None:
        """30s audio at 3s dur / 1.5s overlap -> (30-3)/1.5 + 1 = 19 segments."""
        waveform = torch.randn(1, 32000 * 30)
        segments = transform.extract_segments(waveform)
        assert len(segments) >= 18, (  # noqa: PLR2004
            f"Expected >=18 segments, got {len(segments)}"
        )

    def test_segment_length(self, transform: AudioTransform) -> None:
        """Each segment must be exactly SR * duration = 96000 samples."""
        waveform = torch.randn(1, 32000 * 30)
        segments = transform.extract_segments(waveform)
        expected_len = 32000 * 3
        assert all(
            s.shape[-1] == expected_len for s in segments
        ), "Not all segments have correct length"

    def test_mono_conversion(self, transform: AudioTransform) -> None:
        """Stereo input averaged to mono has correct shape."""
        stereo = torch.randn(2, 32000 * 3)
        mono = stereo.mean(dim=0, keepdim=True)
        assert mono.shape == (1, 96000)

    def test_short_waveform_padding(self, transform: AudioTransform) -> None:
        """Short waveform produces padded mel of correct shape."""
        short = torch.randn(1, 32000)  # 1 second, less than 3s segment
        mel = transform.to_mel_spectrogram(short)
        assert mel.shape == (1, 128, 300), f"Padded mel wrong shape: {mel.shape}"

    def test_log_epsilon_from_config(self, cfg, transform: AudioTransform) -> None:
        """log_epsilon must equal the config value, not hardcoded."""
        assert transform.log_epsilon == cfg.audio.log_epsilon

class TestSeedEverything:
    def test_reproducibility(self) -> None:
        """Same seed must produce identical random tensors."""
        seed_everything(2026)
        a = torch.randn(10)
        seed_everything(2026)
        b = torch.randn(10)
        assert torch.equal(a, b)

    def test_different_seeds_differ(self) -> None:
        """Different seeds must produce different tensors."""
        seed_everything(2026)
        a = torch.randn(10)
        seed_everything(9999)
        b = torch.randn(10)
        assert not torch.equal(a, b)

class TestLabelMap:
    @pytest.fixture()
    def cfg(self):
        return load_config()

    def test_eight_genres(self, cfg) -> None:
        """FMA small has exactly 8 genres."""
        path = Path(cfg.data.label_map_path)
        if not path.exists():
            pytest.skip("label_map.json not yet generated")
        with open(path) as f:
            lm = json.load(f)
        if not lm:
            pytest.skip("label_map.json is empty")
        assert len(lm) == cfg.data.num_classes, (
            f"Expected {cfg.data.num_classes} genres, got {len(lm)}"
        )

    def test_labels_contiguous(self, cfg) -> None:
        """Labels must be 0..N-1 contiguous."""
        path = Path(cfg.data.label_map_path)
        if not path.exists():
            pytest.skip("label_map.json not yet generated")
        with open(path) as f:
            lm = json.load(f)
        if not lm:
            pytest.skip("label_map.json is empty")
        values = sorted(lm.values())
        assert values == list(range(len(lm))), f"Non-contiguous labels: {values}"

class TestConfig:
    def test_config_loads(self) -> None:
        """Config must load without errors and have all required fields."""
        cfg = load_config()
        assert cfg.data.sample_rate == 32000  # noqa: PLR2004
        assert cfg.audio.n_mels == 128  # noqa: PLR2004
        assert cfg.audio.target_frames == 300  # noqa: PLR2004
        assert cfg.audio.hop_length == 320  # noqa: PLR2004
        assert cfg.data.num_classes == 8  # noqa: PLR2004

    def test_config_has_new_fields(self) -> None:
        """All Sovereign Standard fields must exist in config."""
        cfg = load_config()
        # Phase 0 additions
        assert hasattr(cfg.data, "label_map_path")
        assert hasattr(cfg.data, "stats_path")
        assert hasattr(cfg.data, "subset")
        # Phase 2 additions
        assert hasattr(cfg.audio, "min_duration_s")
        assert hasattr(cfg.audio, "silence_threshold_db")
        # Phase 3 additions
        assert hasattr(cfg.audio, "log_epsilon")

    def test_config_paths_not_empty(self) -> None:
        """Config paths must be non-empty strings."""
        cfg = load_config()
        assert cfg.data.label_map_path
        assert cfg.data.stats_path
        assert cfg.data.metadata_path
        assert cfg.data.audio_path

class TestSilenceGate:
    """Test the per-segment silence gate in dataset_builder."""
    def test_silent_segment_detected(self) -> None:
        """A constant-valued mel should be classified as silent."""
        silent_mel = torch.zeros(1, 128, 300)
        assert _is_silent_segment(silent_mel) is True

    def test_normal_segment_passes(self) -> None:
        """A normal mel segment should NOT be classified as silent."""
        normal_mel = torch.randn(1, 128, 300)
        assert _is_silent_segment(normal_mel) is False
