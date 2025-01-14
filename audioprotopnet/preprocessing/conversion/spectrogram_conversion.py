from typing import Optional

import torch
import torchaudio


def convert_spectrogram_to_waveform(
    spectrogram: torch.Tensor,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    n_mels: Optional[int] = None,
) -> torch.Tensor:
    """
    Convert a (mel) spectrogram back to an audio waveform.

    Args:
        spectrogram (torch.Tensor): The input spectrogram as a torch tensor.
        sample_rate (int): The sample rate of the audio.
        n_fft (int): The number of FFT points.
        hop_length (int): The hop length for the STFT calculation.
        n_mels (Optional[int]): The number of Mel filterbanks for the Mel spectrogram. Defaults to None.

    Returns:
        torch.Tensor: The audio waveform.
    """
    device = spectrogram.device  # Get the device of the input spectrogram

    if n_mels:
        # Convert the mel spectrogram to a power spectrogram using torchaudio's inverse mel spectrogram transform
        n_stft = n_fft // 2 + 1
        transform_mel_to_power = torchaudio.transforms.InverseMelScale(
            n_stft=n_stft, n_mels=n_mels, sample_rate=sample_rate, driver="gelsd"
        ).to(
            device
        )  # Move the transform to the same device as the input tensor

        # Convert the power spectrogram back to audio using torchaudio's Griffin-Lim transform
        transform_power_to_waveform = torchaudio.transforms.GriffinLim(
            n_fft=n_fft, hop_length=hop_length
        ).to(
            device
        )  # Move the transform to the same device as the input tensor

        # Apply transformations
        spectrogram = transform_mel_to_power(spectrogram)
        waveform = transform_power_to_waveform(spectrogram)
    else:
        # Convert the power spectrogram back to audio using torchaudio's Griffin-Lim transform
        transform_power_to_waveform = torchaudio.transforms.GriffinLim(
            n_fft=n_fft, hop_length=hop_length
        ).to(
            device
        )  # Move the transform to the same device as the input tensor
        waveform = transform_power_to_waveform(spectrogram)

    return waveform
