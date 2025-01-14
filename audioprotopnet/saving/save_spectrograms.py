import gc
from typing import Optional, Tuple

import cv2
import librosa
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import numpy as np
import numpy.typing as npt


def save_spectrogram_as_figure(
    file_name: str,
    spectrogram: npt.NDArray[np.float32],
    sample_rate: int,
    hop_length: int,
    n_fft: int,
    n_mels: Optional[int],
) -> None:
    """
    Saves a power spectrogram in decibel units as a figure.

    Args:
        file_name (str): The name of the file to save the image as.
        spectrogram (npt.NDArray[np.float32]): The power spectrogram in decibel units to save, represented as a 2D numpy array.
        sample_rate (int): The sampling rate of the audio signal used to compute the spectrogram.
        hop_length (int): The hop length for the STFT calculation.
        n_fft (int): The number of FFT points.
        n_mels (Optional[int]): The number of Mel bands, if using a Mel spectrogram.

    Returns:
        None
    """

    if spectrogram.ndim != 2:
        raise ValueError("Spectrogram must be a 2D array.")

    if hop_length <= 0:
        raise ValueError("Hop length must be a positive integer.")

    if n_fft <= 0:
        raise ValueError("n_fft must be a positive integer.")

    if spectrogram.size == 0:
        raise ValueError("Spectrogram must not be empty.")

    # Determine the y-axis scale
    y_axis = "mel" if n_mels else "linear"

    # Create a figure and axis for the plot
    figure, axes = plt.subplots(nrows=1, ncols=1)

    # Plot the spectrogram
    spectrogram_image = librosa.display.specshow(
        data=spectrogram,
        y_axis=y_axis,
        x_axis="s",
        sr=sample_rate,
        hop_length=hop_length,
        n_fft=n_fft,
        ax=axes,
        cmap="viridis",
    )

    # Add a colorbar to the plot
    figure.colorbar(spectrogram_image, format="%+2.0f dB", ax=axes)

    # Save the plot to a file
    figure.savefig(file_name)

    plt.close(figure)
    gc.collect()


def save_spectrogram_with_bounding_box(
    file_name: str,
    spectrogram: npt.NDArray[np.float32],
    sample_rate: int,
    hop_length: int,
    n_fft: int,
    n_mels: Optional[int],
    bbox_height_start: int,
    bbox_height_end: int,
    bbox_width_start: int,
    bbox_width_end: int,
    color: Tuple[int, int, int] = (255, 0, 0),
    minimalist: bool = False,
) -> None:
    """
    Saves a power spectrogram in decibel units with a bounding box drawn around the specified region.

    Args:
        file_name (str): The name of the file to save the image as.
        spectrogram (npt.NDArray[np.float32]): The power spectrogram in decibel units to save, represented as a 2D numpy array.
        sample_rate (int): The sampling rate of the audio signal used to compute the spectrogram.
        hop_length (int): The hop length for the STFT calculation.
        n_fft (int): The number of FFT points.
        n_mels (Optional[int]): The number of Mel bands, if using a Mel spectrogram.
        bbox_height_start (int): The top y-coordinate of the bounding box.
        bbox_height_end (int): The bottom y-coordinate of the bounding box.
        bbox_width_start (int): The left x-coordinate of the bounding box.
        bbox_width_end (int): The right x-coordinate of the bounding box.
        color (Tuple[int, int, int]): The color of the bounding box. Defaults to (255, 0, 0).

    Returns:
        None
    """

    # Validation checks
    if spectrogram.ndim != 2:
        raise ValueError("Spectrogram must be a 2D array.")

    if hop_length <= 0:
        raise ValueError("Hop length must be a positive integer.")

    if n_fft <= 0:
        raise ValueError("n_fft must be a positive integer.")

    if spectrogram.size == 0:
        raise ValueError("Spectrogram must not be empty.")

    if not (0 <= bbox_height_start < bbox_height_end <= spectrogram.shape[0]):
        raise ValueError("Bounding box height coordinates are invalid.")

    if not (0 <= bbox_width_start < bbox_width_end <= spectrogram.shape[1]):
        raise ValueError("Bounding box width coordinates are invalid.")

    # Normalize the spectrogram values to be between 0 and 1
    min_value_spectrogram = np.min(spectrogram)
    max_value_spectrogram = np.max(spectrogram)
    spectrogram_min_max_normalized = (spectrogram - min_value_spectrogram) / (
        max_value_spectrogram - min_value_spectrogram
    )

    # Apply the colormap to the normalized spectrogram
    colormap = plt.get_cmap("viridis")
    spectrogram_colored = (colormap(spectrogram_min_max_normalized) * 255).astype(
        np.uint8
    )

    # Convert the colored spectrogram to BGR format for OpenCV
    spectrogram_bgr_uint8 = cv2.cvtColor(spectrogram_colored, cv2.COLOR_RGB2BGR)

    # Draw the bounding box on the spectrogram
    # cv2.rectangle(
    #     spectrogram_bgr_uint8,
    #     (bbox_width_start, bbox_height_start),
    #     (bbox_width_end - 1, bbox_height_end - 1),
    #     color,
    #     thickness=2,
    # )

    # Define thickness for horizontal and vertical lines
    horizontal_thickness = 2
    vertical_thickness = 4  # Double the thickness of horizontal lines

    # Top horizontal line
    cv2.line(
        spectrogram_bgr_uint8,
        (bbox_width_start, bbox_height_start),
        (bbox_width_end - 1, bbox_height_start),
        color,
        thickness=horizontal_thickness,
    )

    # Bottom horizontal line
    cv2.line(
        spectrogram_bgr_uint8,
        (bbox_width_start, bbox_height_end - 1),
        (bbox_width_end - 1, bbox_height_end - 1),
        color,
        thickness=horizontal_thickness,
    )

    # Left vertical line
    cv2.line(
        spectrogram_bgr_uint8,
        (bbox_width_start, bbox_height_start),
        (bbox_width_start, bbox_height_end - 1),
        color,
        thickness=vertical_thickness,
    )

    # Right vertical line
    cv2.line(
        spectrogram_bgr_uint8,
        (bbox_width_end - 1, bbox_height_start),
        (bbox_width_end - 1, bbox_height_end - 1),
        color,
        thickness=vertical_thickness,
    )

    # Convert the spectrogram back to RGB format
    spectrogram_rgb_uint8 = spectrogram_bgr_uint8[..., ::-1]

    # Convert image to float and scale values to between 0 and 1
    spectrogram_rgb_float = np.float32(spectrogram_rgb_uint8) / 255

    # Determine the y-axis scale
    y_axis = "mel" if n_mels else "linear"

    # Create a figure and axis for the plot
    figure, axes = plt.subplots(nrows=1, ncols=1)

    # Plot the spectrogram with bounding box
    spectrogram_image = librosa.display.specshow(
        data=spectrogram_rgb_float,
        y_axis=y_axis,
        x_axis="s",
        sr=sample_rate,
        hop_length=hop_length,
        n_fft=n_fft,
        ax=axes,
    )

    if not minimalist:
        # Create a ScalarMappable for the colorbar
        sm = ScalarMappable(
            cmap="viridis",
            norm=plt.Normalize(vmin=min_value_spectrogram, vmax=max_value_spectrogram),
        )

        # Add a colorbar to the plot
        sm.set_array([])
        figure.colorbar(sm, format="%+2.0f dB", ax=axes)
    else:
        # Remove axis labels and ticks
        axes.set_xticks([])
        axes.set_yticks([])
        axes.set_xlabel("")
        axes.set_ylabel("")
        axes.set_title("")

    # Save the plot to a file
    figure.savefig(file_name)

    plt.close(figure)
    gc.collect()


def save_spectrogram_with_heatmap(
    file_name: str,
    spectrogram: npt.NDArray[np.float32],
    activation_pattern: npt.NDArray[np.float32],
    sample_rate: int,
    hop_length: int,
    n_fft: int,
    n_mels: Optional[int],
) -> None:
    """
    Saves a power spectrogram in decibel units with a heatmap overlay based on the activation pattern.

    Args:
        file_name (str): The name of the file to save the image as.
        spectrogram (npt.NDArray[np.float32]): The power spectrogram in decibel units to save, represented as a 2D numpy array.
        activation_pattern (npt.NDArray[np.float32]): The activation pattern used to generate the heatmap overlay, represented as a 2D numpy array.
        sample_rate (int): The sampling rate of the audio signal used to compute the spectrogram.
        hop_length (int): The hop length for the STFT calculation.
        n_fft (int): The number of FFT points.
        n_mels (Optional[int]): The number of Mel bands, if using a Mel spectrogram.

    Returns:
        None
    """

    # Validation checks
    if spectrogram.ndim != 2:
        raise ValueError("Spectrogram must be a 2D array.")

    if activation_pattern.ndim != 2:
        raise ValueError("Activation pattern must be a 2D array.")

    if hop_length <= 0:
        raise ValueError("Hop length must be a positive integer.")

    if n_fft <= 0:
        raise ValueError("n_fft must be a positive integer.")

    if spectrogram.size == 0:
        raise ValueError("Spectrogram must not be empty.")

    if activation_pattern.size == 0:
        raise ValueError("Activation pattern must not be empty.")

    # Normalize the spectrogram values to be between 0 and 1
    min_value_spectrogram = np.min(spectrogram)
    max_value_spectrogram = np.max(spectrogram)
    spectrogram_min_max_normalized = (spectrogram - min_value_spectrogram) / (
        max_value_spectrogram - min_value_spectrogram
    )

    # Apply the colormap to the normalized spectrogram
    colormap = plt.get_cmap("viridis")
    spectrogram_colored = colormap(spectrogram_min_max_normalized)

    # Resize the activation pattern to match the spectrogram dimensions
    spectrogram_height, spectrogram_width = spectrogram.shape
    upsampled_activation_pattern = cv2.resize(
        activation_pattern,
        dsize=(spectrogram_width, spectrogram_height),
        interpolation=cv2.INTER_CUBIC,
    )

    # Normalize the activation pattern
    rescaled_activation_pattern = upsampled_activation_pattern - np.amin(
        upsampled_activation_pattern
    )
    rescaled_activation_pattern /= np.amax(rescaled_activation_pattern)

    # Create the heatmap
    colormap_heatmap = plt.get_cmap("Reds")
    heatmap = colormap_heatmap(rescaled_activation_pattern)

    # Combine the spectrogram and heatmap
    spectrogram_with_heatmap = 0.3 * spectrogram_colored + 0.7 * heatmap

    # Determine the y-axis scale
    y_axis = "mel" if n_mels else "linear"

    # Create a figure and axis for the plot
    figure, axes = plt.subplots(nrows=1, ncols=1)

    # Plot the spectrogram with heatmap
    spectrogram_image = librosa.display.specshow(
        data=spectrogram_with_heatmap,
        y_axis=y_axis,
        x_axis="s",
        sr=sample_rate,
        hop_length=hop_length,
        n_fft=n_fft,
        ax=axes,
    )

    # Create a ScalarMappable for the colorbar
    min_value_activation_pattern = np.min(activation_pattern)
    max_value_activation_pattern = np.max(activation_pattern)
    sm = ScalarMappable(
        cmap="Reds",
        norm=plt.Normalize(
            vmin=min_value_activation_pattern, vmax=max_value_activation_pattern
        ),
    )

    # Add a colorbar to the plot
    sm.set_array([])
    figure.colorbar(sm, ax=axes)

    # Save the plot to a file
    figure.savefig(file_name)

    plt.close(figure)
    gc.collect()
