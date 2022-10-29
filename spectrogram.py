import torch
import torch.nn as nn
from nnAudio import features


class MelSpectrogram(nn.Module):
    def __init__(self, sample_rate: int, n_ftt: int, n_mels: int, hop_length: int):
        """
        Melspectrogram transformation layer, supports on-the-fly processing on GPU.

        Attributes:
            sample_rate: The sampling rate for the input audio.
            n_ftt: The window size for the STFT.
            n_mels: The number of Mel filter banks.
            hop_length: The hop (or stride) size.
        """
        super().__init__()
        self.transform = features.MelSpectrogram(
            sr=sample_rate,
            n_fft=n_ftt,
            n_mels=n_mels,
            hop_length=hop_length,
            center=True,
            fmin=0,
            fmax=sample_rate // 2,
            pad_mode="constant",
        )

    def forward(self, samples: torch.tensor) -> torch.tensor:
        """
        Convert a batch of audio frames into a batch of Mel spectrogram frames.

        For each item in the batch:
        1. pad left and right ends of audio by n_fft // 2.
        2. run STFT with window size of |n_ftt| and stride of |hop_length|.
        3. convert result into mel-scale.
        4. therefore, n_frames = n_samples // hop_length + 1.

        Args:
            samples: Audio time-series (batch size, n_samples).

        Returns:
            A batch of Mel spectrograms of size (batch size, n_frames, n_mels).
        """
        spectrogram = self.transform(samples)
        spectrogram = spectrogram.permute(0, 2, 1)
        return spectrogram
