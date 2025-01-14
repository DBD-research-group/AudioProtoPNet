import logging
import os
from typing import Optional, Tuple

import numpy as np
import torch

from audioprotopnet.saving.save_spectrograms import (
    save_spectrogram_as_figure,
    save_spectrogram_with_bounding_box,
    save_spectrogram_with_heatmap,
)
from audioprotopnet.saving.save_waveforms import (
    save_prototype_as_waveform,
    save_spectrogram_as_waveform,
)


def save_prototype_files(
    spectrogram_unnormalized_torch: torch.Tensor,
    spectrogram_unnormalized_np: np.ndarray,
    bounding_box_high_activation_patch: Tuple[int, int, int, int],
    activation_pattern: np.ndarray,
    local_analysis_dir: str,
    logger: logging.Logger,
    sample_rate: int,
    hop_length: int,
    n_fft: int,
    n_mels: Optional[int],
    save_prototype_waveform_files: bool,
    save_prototype_spectrogram_files: bool,
    prototype_rank: Optional[int] = None,
    class_index: Optional[int] = None,
    prototype_count: Optional[int] = None,
) -> None:
    """
    Save visualizations related to a prototype and log information.

    This function saves visualizations for either the most highly activated prototype or
    a top class prototype, depending on the provided parameters.

    Args:
        spectrogram_unnormalized_torch (torch.Tensor): Unnormalized spectrogram in PyTorch tensor format.
        spectrogram_unnormalized_np (np.ndarray): Unnormalized spectrogram in numpy array format without channel dimension, i.e., shape (H, W).
        bounding_box_high_activation_patch (Tuple[int, int, int, int]): Bounding box coordinates of the highly activated patch.
        activation_pattern (np.ndarray): Activation pattern of the prototype.
        local_analysis_dir (str): Directory where the visualizations will be saved.
        logger (logging.Logger): Logger for logging information.
        sample_rate (int): Sample rate of the audio.
        hop_length (int): Hop length used in spectrogram computation.
        n_fft (int): Number of FFT components.
        n_mels (Optional[int]): Number of Mel bands.
        save_prototype_waveform_files (bool): Whether to save the prototype waveform files.
        save_prototype_spectrogram_files (bool): Whether to save the prototype spectrogram files.
        prototype_rank (Optional[int]): Rank of the prototype in activation analysis (for most activated prototype).
        class_index (Optional[int]): Index of the top predicted class (for top class prototype).
        prototype_count (Optional[int]): Count of the prototype within its class (for top class prototype).

    Raises:
        ValueError: If neither prototype_rank nor both class_index and prototype_count are provided.
    """

    # Determine the sub-directory and file suffix based on the given parameters
    if prototype_rank is not None:
        sub_dir = "most-activated-prototypes"
        file_suffix = f"top-{prototype_rank}-prototype"
    elif class_index is not None and prototype_count is not None:
        sub_dir = f"top-{class_index + 1}-class-prototypes"
        file_suffix = f"top-{prototype_count}-prototype"
    else:
        raise ValueError(
            "Either prototype_rank or both class_index and prototype_count must be provided."
        )

    # Save the waveform of the highly activated patch if requested
    if save_prototype_waveform_files:
        logger.info(
            "Saving the most highly activated patch of the current sample's spectrogram as waveform..."
        )
        save_prototype_as_waveform(
            spectrogram=spectrogram_unnormalized_torch,
            bbox_height_start=bounding_box_high_activation_patch[0],
            bbox_height_end=bounding_box_high_activation_patch[1],
            bbox_width_start=bounding_box_high_activation_patch[2],
            bbox_width_end=bounding_box_high_activation_patch[3],
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            db_scale=True,
            normalized=False,
            mean=None,
            std=None,
            output_file=os.path.join(
                local_analysis_dir,
                sub_dir,
                f"most-highly-activated-patch-by-{file_suffix}_waveform.wav",
            ),
        )

    # Save the spectrogram visualizations if requested
    if save_prototype_spectrogram_files:
        logger.info(
            "Saving the prototype activation map of the current sample's spectrogram..."
        )

        # Save the spectrogram with an overlayed heatmap of the activation pattern
        save_spectrogram_with_heatmap(
            file_name=os.path.join(
                local_analysis_dir,
                sub_dir,
                f"prototype-activation-map-by-{file_suffix}.pdf",
            ),
            spectrogram=spectrogram_unnormalized_np,
            activation_pattern=activation_pattern,
            sample_rate=sample_rate,
            hop_length=hop_length,
            n_fft=n_fft,
            n_mels=n_mels,
        )

        logger.info(
            "Saving the spectrogram with the bounding box of the most highly activated patch..."
        )

        # Save the spectrogram with a bounding box around the highly activated patch
        save_spectrogram_with_bounding_box(
            file_name=os.path.join(
                local_analysis_dir,
                sub_dir,
                f"most-highly-activated-patch-in-original-spectrogram-by-{file_suffix}.pdf",
            ),
            spectrogram=spectrogram_unnormalized_np,
            sample_rate=sample_rate,
            hop_length=hop_length,
            n_fft=n_fft,
            n_mels=n_mels,
            bbox_height_start=bounding_box_high_activation_patch[0],
            bbox_height_end=bounding_box_high_activation_patch[1],
            bbox_width_start=bounding_box_high_activation_patch[2],
            bbox_width_end=bounding_box_high_activation_patch[3],
            color=(21, 101, 249),
        )


def save_sample_files(
    spectrogram_np: np.ndarray,
    spectrogram_torch: torch.Tensor,
    local_analysis_dir: str,
    sample_rate: int,
    hop_length: int,
    n_fft: int,
    n_mels: Optional[int],
    save_prototype_waveform_files: bool,
    save_prototype_spectrogram_files: bool,
) -> None:
    """
    Save the unnormalized power spectrogram of a sample as both a figure and a waveform.

    Args:
        spectrogram_np (np.ndarray): Unnormalized spectrogram in numpy array format (H, W).
        spectrogram_torch (torch.Tensor): Unnormalized spectrogram in PyTorch tensor format (N, C, H, W).
        local_analysis_dir (str): Path to save the spectrogram visualizations.
        sample_rate (int): Sample rate of the audio.
        hop_length (int): Hop length used in spectrogram computation.
        n_fft (int): Number of FFT components.
        n_mels (Optional[int]): Number of Mel bands.
        save_prototype_waveform_files (bool): Whether to save the spectrogram as a waveform.
        save_prototype_spectrogram_files (bool): Whether to save the spectrogram as a figure.
    """

    # Save the spectrogram as a figure in decibel units if requested
    if save_prototype_spectrogram_files:
        save_spectrogram_as_figure(
            file_name=os.path.join(local_analysis_dir, "original-spectrogram.pdf"),
            spectrogram=spectrogram_np,
            sample_rate=sample_rate,
            hop_length=hop_length,
            n_fft=n_fft,
            n_mels=n_mels,
        )

    # Save the spectrogram as an audio waveform if requested
    if save_prototype_waveform_files:
        save_spectrogram_as_waveform(
            spectrogram=spectrogram_torch,
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            db_scale=True,
            normalized=False,
            mean=None,
            std=None,
            output_file=os.path.join(local_analysis_dir, "original-waveform.wav"),
        )
