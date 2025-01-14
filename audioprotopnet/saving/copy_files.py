import logging
import os
import shutil
from typing import Optional

# import torchaudio


# def copy_prototype_spectrogram(
#     file_name: str,
#     index: int,
#     category: str,
#     log: logging.Logger,
#     load_prototype_files_dir: str,
# ) -> None:
#     """
#     Copy a prototype spectrogram PDF to a new location.
#
#     Args:
#         file_name (str): The name of the file to save the PDF as.
#         index (int): The index of the prototype.
#         category (str): The prototype label in string format.
#         log (logging.Logger): The logger used for logging.
#         load_prototype_files_dir (str): The directory path where the prototype PDF is located.
#
#     Returns:
#         None
#
#     Raises:
#         FileNotFoundError: If the prototype PDF file does not exist at the given path.
#         IOError: If the prototype PDF file cannot be copied for reasons other than non-existence.
#         Exception: Unexpected error while processing the prototype PDF file.
#
#     Example:
#         logger = logging.getLogger(__name__)
#         copy_prototype_spectrogram('new_image.pdf', 10, 'category_name', logger, '/path/to/images/')
#     """
#     # Construct the full path to the prototype PDF file
#     file_path = os.path.join(load_prototype_files_dir, category, str(index))
#     pdf_path = os.path.join(
#         file_path, f"prototype_{category}_{index}_original_bounding-box.pdf"
#     )
#     output_path = os.path.join(os.getcwd(), file_name)
#
#     try:
#         # Attempt to copy the prototype PDF to the new location
#         shutil.copy(pdf_path, output_path)
#     except FileNotFoundError:
#         log.error(f"File {pdf_path} not found.")
#     except IOError:
#         log.error(f"IOError while copying the file {pdf_path}.")
#     except Exception as error:
#         log.error(
#             f"Unexpected error while copying {pdf_path}. An exception occurred: {error}"
#         )
#
#
# def copy_prototype_waveform(
#     file_name: str,
#     category: str,
#     index: int,
#     log: logging.Logger,
#     load_prototype_files_dir: str,
# ) -> None:
#     """
#     Save a prototype waveform to a file.
#
#     Args:
#         file_name (str): The name of the file to save the waveform as.
#         category (str): The prototype label in string format.
#         index (int): The index of the prototype.
#         log (logging.Logger): The logger used for logging.
#         load_prototype_files_dir (str): The directory path where the prototype waveform is located.
#
#     Returns:
#         None
#
#     Raises:
#         FileNotFoundError: If the waveform .wav file does not exist at the given path.
#         IOError: If the waveform file cannot be loaded for reasons other than non-existence.
#         Exception: Unexpected error while processing waveform.
#
#     Example:
#         logger = logging.getLogger(__name__)
#         copy_prototype_waveform('audio.wav', 'category_name', 10, logger, '/path/to/audios/')
#     """
#     # Construct the full path to the prototype waveform file
#     file_path = os.path.join(load_prototype_files_dir, category, str(index))
#     waveform_path = os.path.join(
#         file_path, f"prototype_{category}_{index}_part_waveform.wav"
#     )
#
#     try:
#         # Load the audio waveform of the prototype
#         prototype_waveform, sample_rate = torchaudio.load(waveform_path)
#
#         # Save the audio waveform of the prototype to a file
#         torchaudio.save(file_name, prototype_waveform, sample_rate)
#     except FileNotFoundError:
#         log.error(f"File {waveform_path} not found.")
#     except IOError:
#         log.error(f"IOError while handling the file {waveform_path}.")
#     except Exception as error:
#         log.error(
#             f"Unexpected error while processing {waveform_path}. An exception occurred: {error}"
#         )


def copy_prototype_folder(
    index: int,
    category: str,
    log: logging.Logger,
    load_prototype_files_dir: str,
    destination_dir: str,
) -> None:
    """
    Copy a prototype folder to a new location.

    Args:
        index (int): The index of the prototype.
        category (str): The prototype label in string format.
        log (logging.Logger): The logger used for logging.
        load_prototype_files_dir (str): The directory path where the prototype folder is located.
        destination_dir (str): The directory path where the prototype folder should be copied to.

    Returns:
        None
    """
    # Construct the full path to the prototype folder
    src_folder_path = os.path.join(load_prototype_files_dir, category, str(index))
    dst_folder_path = os.path.join(destination_dir, f"prototype_{category}_{index}")

    try:
        # If the destination directory exists, remove it first
        if os.path.exists(dst_folder_path):
            shutil.rmtree(dst_folder_path)

        # Attempt to copy the prototype folder to the new location
        shutil.copytree(src_folder_path, dst_folder_path)
        log.info(
            f"Successfully copied folder from {src_folder_path} to {dst_folder_path}."
        )
    except FileNotFoundError:
        log.error(f"Folder {src_folder_path} not found.")
    except IOError:
        log.error(f"IOError while copying the folder {src_folder_path}.")
    except Exception as error:
        log.error(
            f"Unexpected error while copying {src_folder_path}. An exception occurred: {error}"
        )


def copy_prototype_files(
    prototype_index_original: int,
    prototype_category: str,
    local_analysis_dir: str,
    prototype_files_dir: str,
    logger: logging.Logger,
    prototype_rank: Optional[int] = None,
    class_index: Optional[int] = None,
    prototype_count: Optional[int] = None,
) -> None:
    """
    Copies the entire prototype folder to a new location.

    Args:
        prototype_index_original (int): The index of the prototype.
        prototype_category (str): The category of the prototype.
        local_analysis_dir (str): Directory path where the prototype folder will be copied to.
        prototype_files_dir (str): Directory where the prototype folders are stored.
        logger (logging.Logger): Logger for logging information.
        prototype_rank (Optional[int]): The rank of the prototype among the most activated prototypes. Used if class_index is None.
        class_index (Optional[int]): The index of the top predicted class. Used if prototype_rank is None.
        prototype_count (Optional[int]): The count of the prototype within its class. Used if class_index is not None.

    Returns:
        None
    """
    if class_index is not None and prototype_count is not None:
        # Determine the subdirectory for top class prototypes
        sub_dir = f"top-{class_index + 1}-class-prototypes"
    elif prototype_rank is not None:
        # Determine the subdirectory for most activated prototypes
        sub_dir = "most-activated-prototypes"
    else:
        # Raise an error if required parameters are not provided
        raise ValueError(
            "Either prototype_rank or class_index with prototype_count must be provided."
        )

    # Construct the full path for the destination directory
    destination_dir = os.path.join(local_analysis_dir, sub_dir)
    os.makedirs(destination_dir, exist_ok=True)

    # Copy the entire prototype folder
    copy_prototype_folder(
        index=prototype_index_original,
        category=prototype_category,
        log=logger,
        load_prototype_files_dir=prototype_files_dir,
        destination_dir=destination_dir,
    )


# def copy_prototype_files(
#     prototype_index_original: int,
#     prototype_category: str,
#     local_analysis_dir: str,
#     prototype_files_dir: str,
#     logger: logging.Logger,
#     save_prototype_waveform_files: bool,
#     save_prototype_spectrogram_files: bool,
#     prototype_rank: Optional[int] = None,
#     class_index: Optional[int] = None,
#     prototype_count: Optional[int] = None,
# ) -> None:
#     """
#     Saves various visualizations related to a prototype as both spectrogram and waveform.
#
#     This function can be used to save files for the most activated prototypes or top class prototypes
#     depending on the provided parameters.
#
#     Args:
#         prototype_index_original (int): The index of the prototype.
#         prototype_category (str): The category of the prototype.
#         local_analysis_dir (str): Directory path where the visualizations will be saved.
#         prototype_files_dir (str): Directory where the prototype files are stored.
#         logger (logging.Logger): Logger for logging information.
#         save_prototype_waveform_files (bool): Flag to indicate if prototype waveforms should be saved.
#         save_prototype_spectrogram_files (bool): Flag to indicate if prototype spectrograms should be saved.
#         prototype_rank (Optional[int]): The rank of the prototype among the most activated prototypes. Used if class_index is None.
#         class_index (Optional[int]): The index of the top predicted class. Used if prototype_rank is None.
#         prototype_count (Optional[int]): The count of the prototype within its class. Used if class_index is not None.
#
#     Returns:
#         None
#
#     Raises:
#         ValueError: If neither prototype_rank nor class_index with prototype_count are provided.
#     """
#     if class_index is not None and prototype_count is not None:
#         # Determine the subdirectory and file suffixes for top class prototypes
#         sub_dir = f"top-{class_index + 1}-class-prototypes"
#         spectrogram_suffix = (
#             f"top-{prototype_count}-activated-prototype_original_bounding-box.pdf"
#         )
#         waveform_suffix = f"top-{prototype_count}-activated-prototype_part_waveform.wav"
#     elif prototype_rank is not None:
#         # Determine the subdirectory and file suffixes for most activated prototypes
#         sub_dir = "most-activated-prototypes"
#         spectrogram_suffix = (
#             f"top-{prototype_rank}-activated-prototype_original_bounding-box.pdf"
#         )
#         waveform_suffix = f"top-{prototype_rank}-activated-prototype_part_waveform.wav"
#     else:
#         # Raise an error if required parameters are not provided
#         raise ValueError(
#             "Either prototype_rank or class_index with prototype_count must be provided."
#         )
#
#     # Construct the full path for the output files
#     spectrogram_file_name = os.path.join(
#         local_analysis_dir, sub_dir, spectrogram_suffix
#     )
#     waveform_file_name = os.path.join(local_analysis_dir, sub_dir, waveform_suffix)
#
#     if save_prototype_spectrogram_files:
#         # Save the prototype as an unnormalized power spectrogram in dB scale
#         copy_prototype_spectrogram(
#             file_name=spectrogram_file_name,
#             index=prototype_index_original,
#             category=prototype_category,
#             log=logger,
#             load_prototype_files_dir=prototype_files_dir,
#         )
#
#     if save_prototype_waveform_files:
#         # Save the prototype as a waveform
#         copy_prototype_waveform(
#             file_name=waveform_file_name,
#             index=prototype_index_original,
#             category=prototype_category,
#             log=logger,
#             load_prototype_files_dir=prototype_files_dir,
#         )
