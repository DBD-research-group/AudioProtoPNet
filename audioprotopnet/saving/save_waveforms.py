from typing import Optional

import librosa
import torch
import torchaudio

from audioprotopnet.preprocessing.conversion.spectrogram_conversion import convert_spectrogram_to_waveform
from audioprotopnet.preprocessing.normalization import undo_standardize_tensor


def save_spectrogram_as_waveform(
    spectrogram: torch.Tensor,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    n_mels: Optional[int] = None,
    db_scale: bool = False,
    normalized: bool = False,
    mean: Optional[float] = None,
    std: Optional[float] = None,
    output_file: str = "output_audio.wav",
) -> None:
    """
    Save a (mel) spectrogram as an audio file.

    Args:
        spectrogram (torch.Tensor): The input spectrogram as a torch tensor.
        sample_rate (int): The sample rate of the audio.
        n_fft (int): The number of FFT points.
        hop_length (int): The hop length for the STFT calculation.
        n_mels (Optional[int]): The number of Mel filterbanks for the Mel spectrogram. Defaults to None.
        db_scale (bool): Flag indicating whether the input spectrogram is in decibel (dB) units.
        normalized (bool): Whether the input spectrogram is normalized. Defaults to False.
        mean (Optional[float]): The mean value for undoing z-standardization. Only needed if normalized=True.
        std (Optional[float]): The standard deviation for undoing z-standardization. Only needed if normalized=True.
        output_file (str): The path to save the output audio file. Defaults to "output_audio.wav".

    Returns:
        None
    """

    if not isinstance(spectrogram, torch.Tensor):
        raise TypeError("Spectrogram must be a torch.Tensor.")

    # Clone the input spectrogram to avoid modifying the original data
    spectrogram = spectrogram.clone()

    if normalized:
        # Undo z-standardization if the spectrogram is normalized
        spectrogram = undo_standardize_tensor(x=spectrogram, mean=mean, std=std)

    if db_scale:
        # Convert the spectrogram from dB scale to power scale
        spectrogram = spectrogram.cpu().numpy()
        spectrogram = librosa.db_to_power(spectrogram)
        spectrogram = torch.from_numpy(spectrogram)

    # Convert the (mel) spectrogram back to waveform
    waveform = convert_spectrogram_to_waveform(
        spectrogram, sample_rate, n_fft, hop_length, n_mels
    )

    # Squeeze batch dimension if present
    waveform = waveform.squeeze(0)

    try:
        # Save the audio waveform to a file
        torchaudio.save(output_file, waveform, sample_rate)
    except Exception as e:
        raise FileNotFoundError(f"Failed to save waveform to {output_file}: {e}")


def save_prototype_as_waveform(
    spectrogram: torch.Tensor,
    bbox_height_start: int,
    bbox_height_end: int,
    bbox_width_start: int,
    bbox_width_end: int,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    n_mels: Optional[int] = None,
    db_scale: bool = False,
    normalized: bool = False,
    mean: Optional[float] = None,
    std: Optional[float] = None,
    mask_value: float = 0.0,
    output_file: str = "output_audio.wav",
) -> None:
    """
    Save a bounding box of a spectrogram as an audio file.

    Args:
        spectrogram (torch.Tensor): The input spectrogram as a torch tensor.
        bbox_height_start (int): The start index of the bounding box in the height dimension.
        bbox_height_end (int): The end index of the bounding box in the height dimension.
        bbox_width_start (int): The start index of the bounding box in the width dimension.
        bbox_width_end (int): The end index of the bounding box in the width dimension.
        sample_rate (int): The sample rate of the audio.
        n_fft (int): The number of FFT points.
        hop_length (int): The hop length for the STFT calculation.
        n_mels (Optional[int]): The number of Mel filterbanks for the Mel spectrogram. Defaults to None.
        db_scale (bool): Flag indicating whether the input spectrogram is in decibel (dB) units.
        normalized (bool): Whether the input spectrogram is normalized. Defaults to False.
        mean (Optional[float]): The mean value for undoing z-standardization. Only needed if normalized=True.
        std (Optional[float]): The standard deviation for undoing z-standardization. Only needed if normalized=True.
        mask_value (float): The value to replace outside the bounding box region. Defaults to 0.0.
        output_file (str): The path to save the output audio file. Defaults to "output_audio.wav".

    Returns:
        None
    """
    if not isinstance(spectrogram, torch.Tensor):
        raise TypeError("Spectrogram must be a torch.Tensor")

    if sample_rate <= 0:
        raise ValueError("Sample rate must be a positive integer.")

    if bbox_height_end <= bbox_height_start or bbox_width_end <= bbox_width_start:
        raise RuntimeError("Invalid bounding box coordinates.")

    if normalized and (mean is None or std is None):
        raise TypeError("Mean and std must be provided for normalization.")

    # Clone the input spectrogram to avoid modifying the original data
    masked_spectrogram = spectrogram.clone()

    # Crop the spectrogram to the bounding box region in the width dimension
    masked_spectrogram = masked_spectrogram[:, :, :, bbox_width_start:bbox_width_end]

    if normalized:
        # Undo z-standardization if the spectrogram is normalized
        masked_spectrogram = undo_standardize_tensor(
            x=masked_spectrogram, mean=mean, std=std
        )

    if db_scale:
        # Convert the spectrogram from dB scale to power scale
        masked_spectrogram = masked_spectrogram.cpu().numpy()
        masked_spectrogram = librosa.db_to_power(masked_spectrogram)
        masked_spectrogram = torch.from_numpy(masked_spectrogram)

    # Set values outside the bounding box to the specified mask value
    masked_spectrogram[:, :, :bbox_height_start, :] = mask_value
    masked_spectrogram[:, :, bbox_height_end:, :] = mask_value

    # Save the masked spectrogram as an audio waveform
    save_spectrogram_as_waveform(
        spectrogram=masked_spectrogram,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        db_scale=False,
        normalized=False,
        mean=None,
        std=None,
        output_file=output_file,
    )
