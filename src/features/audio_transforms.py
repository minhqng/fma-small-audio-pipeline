"""
GPU-ready audio processing pipeline.
Uses PyAV for audio loading and torchaudio for MelSpectrogram transforms.
Produces Log-Mel-Spectrograms of shape (1, n_mels, target_frames).
"""
from __future__ import annotations

from dataclasses import dataclass

import av
import numpy as np
import torch
import torchaudio
from omegaconf import DictConfig


def _load_audio_pyav(file_path: str) -> tuple[torch.Tensor, int]:
    """
    Load audio file using PyAV (FFmpeg-based).

    Returns:
        (waveform, sample_rate) where waveform is (channels, samples).
    """
    container = av.open(file_path)
    stream = container.streams.audio[0]
    sr = stream.rate

    frames: list[np.ndarray] = []
    for frame in container.decode(audio=0):
        arr = frame.to_ndarray()  # (channels, samples) float32
        frames.append(arr)
    container.close()

    if not frames:
        msg = f"No audio frames decoded from {file_path}"
        raise RuntimeError(msg)

    audio = np.concatenate(frames, axis=1)  # (channels, total_samples)
    waveform = torch.from_numpy(audio.copy()).float()
    return waveform, sr


@dataclass
class AudioTransform:
    """
    Encapsulates the full audio -> mel-spectrogram pipeline.

    Pipeline per track:
        load MP3 -> mono -> resample to target SR
        -> slice into fixed-length segments with overlap
        -> MelSpectrogram -> log scale
        -> pad/truncate to exact (1, n_mels, target_frames)
    """

    sample_rate: int
    duration: float
    segment_overlap: float
    n_mels: int
    n_fft: int
    hop_length: int
    f_min: int
    f_max: int
    norm_type: str
    target_frames: int
    log_epsilon: float
    device: torch.device

    @classmethod
    def from_config(cls, cfg: DictConfig, device: torch.device) -> AudioTransform:
        """Factory from Hydra config."""
        return cls(
            sample_rate=cfg.data.sample_rate,
            duration=cfg.audio.duration,
            segment_overlap=cfg.audio.segment_overlap,
            n_mels=cfg.audio.n_mels,
            n_fft=cfg.audio.n_fft,
            hop_length=cfg.audio.hop_length,
            f_min=cfg.audio.f_min,
            f_max=cfg.audio.f_max,
            norm_type=cfg.audio.norm_type,
            target_frames=cfg.audio.target_frames,
            log_epsilon=cfg.audio.log_epsilon,
            device=device,
        )

    def _get_mel_transform(self) -> torchaudio.transforms.MelSpectrogram:
        """Create MelSpectrogram transform on target device."""
        return torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            f_min=self.f_min,
            f_max=self.f_max,
            norm=self.norm_type,
            mel_scale="slaney",
        ).to(self.device)

    def load_and_resample(self, file_path: str) -> torch.Tensor:
        """
        Load audio file via PyAV, convert to mono, resample.

        Returns:
            Tensor of shape (1, num_samples)
        """
        waveform, sr = _load_audio_pyav(file_path)

        # Convert to mono by averaging channels
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=self.sample_rate,
            )
            waveform = resampler(waveform)

        return waveform  # (1, num_samples)

    def extract_segments(self, waveform: torch.Tensor) -> list[torch.Tensor]:
        """
        Slice waveform into fixed-length segments with overlap.

        Each segment: (1, segment_samples) where segment_samples = SR * duration.
        Last partial segment is zero-padded only if remaining > 1 second.
        """
        segment_samples = int(self.sample_rate * self.duration)
        hop_samples = int(
            self.sample_rate * (self.duration - self.segment_overlap)
        )
        total_samples = waveform.shape[-1]

        segments: list[torch.Tensor] = []
        start = 0
        while start + segment_samples <= total_samples:
            segment = waveform[:, start : start + segment_samples]
            segments.append(segment)
            start += hop_samples

        # Handle last partial segment by zero-padding
        if start < total_samples and (total_samples - start) > self.sample_rate:
            # Only pad if remaining > 1 second
            last_segment = waveform[:, start:]
            pad_len = segment_samples - last_segment.shape[-1]
            last_segment = torch.nn.functional.pad(last_segment, (0, pad_len))
            segments.append(last_segment)

        return segments

    def to_mel_spectrogram(self, segment: torch.Tensor) -> torch.Tensor:
        """
        Convert waveform segment to log-Mel spectrogram.

        Input:  (1, segment_samples)
        Output: (1, n_mels, target_frames) -- e.g. (1, 128, 300)
        """
        mel_transform = self._get_mel_transform()
        segment = segment.to(self.device)
        mel = mel_transform(segment)  # (1, n_mels, frames)

        # Log scale (config-driven epsilon to avoid log(0))
        mel = torch.log(mel + self.log_epsilon)

        # Truncate or pad to exact target_frames
        if mel.shape[-1] > self.target_frames:
            mel = mel[:, :, : self.target_frames]
        elif mel.shape[-1] < self.target_frames:
            pad_len = self.target_frames - mel.shape[-1]
            mel = torch.nn.functional.pad(mel, (0, pad_len))

        return mel  # (1, 128, 300)

    def process_track(self, file_path: str) -> list[torch.Tensor]:
        """
        Full pipeline for one track:
        load -> resample -> mono -> segment -> mel-spectrogram.

        Returns:
            List of tensors, each of shape (1, n_mels, target_frames).
        """
        waveform = self.load_and_resample(file_path)
        segments = self.extract_segments(waveform)
        return [self.to_mel_spectrogram(seg) for seg in segments]
