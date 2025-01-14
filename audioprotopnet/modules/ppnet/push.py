import gc
import logging
import os
import time
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
import numpy.typing as npt
from scipy import ndimage
import torch

from audioprotopnet.modules.ppnet.ppnet import PPNet
from audioprotopnet.preprocessing.normalization import standardize_tensor
from audioprotopnet.saving.save_spectrograms import (
    save_spectrogram_as_figure,
    save_spectrogram_with_bounding_box,
    save_spectrogram_with_heatmap,
)
from audioprotopnet.saving.save_waveforms import (
    save_prototype_as_waveform,
    save_spectrogram_as_waveform,
)


def find_high_activation_crop(
    activation_map: npt.NDArray[np.float64],
    percentile: float = 95,
) -> Tuple[int, int, int, int]:
    """Calculates the indices of a bounding box for the most highly activated region in an activation map.

    The function calculates a threshold based on the specified percentile of the activation map.
    It then creates a binary mask in which all values in the activation map above the threshold are set to True and all
    other values are set to False. The function then labels connected regions in the binary mask and measures the
    properties of these regions based on the original activation map intensities. It identifies the region with the
    maximum mean intensity as the most highly activated region. The function then extracts the bounding box coordinates
    of this region, defining the spatial extent of the highest activated region in the activation map.

    Args:
        activation_map (npt.NDArray[np.float64]): A two-dimensional activation map for which the most activated patch
                                                  is to be determined.
        percentile (float): The percentile (between 0 and 100) used to calculate a threshold that determines which
                            values of the activation map are considered high activations and which are considered low
                            activations.

    Returns:
        Tuple[int, int, int, int]: The lower and upper row and column indices of the bounding box.

    """
    # Calculate the threshold value based on the specified percentile
    threshold = np.percentile(activation_map, percentile)

    # Create a binary mask where high activations are set to True and low activations to False
    binary_mask = activation_map >= threshold

    # Create a structuring element for connectivity that will consider features connected even if they touch diagonally.
    structure = ndimage.generate_binary_structure(2, 2)

    # Label connected regions in the binary mask
    labeled_mask, num_labels = ndimage.label(input=binary_mask, structure=structure)

    # Calculate mean intensity for each label
    mean_intensities = ndimage.mean(
        activation_map, labeled_mask, range(1, num_labels + 1)
    )

    # Identify the label with the maximum mean intensity
    max_label = np.argmax(mean_intensities) + 1

    # Extract bounding box coordinates for the identified label
    slice_y, slice_x = ndimage.find_objects(labeled_mask == max_label)[0]
    lower_y, upper_y = slice_y.start, slice_y.stop
    lower_x, upper_x = slice_x.start, slice_x.stop

    return lower_y, upper_y, lower_x, upper_x


def stack_prototype_activations(activations):
    # Find the shape of the first non-None array
    valid_shape = None
    for arr in activations:
        if arr is not None:
            valid_shape = arr.shape
            break

    if valid_shape is None:
        raise ValueError(
            "All prototype activations are None, cannot determine a valid shape."
        )

    # Convert None values to arrays of the found valid shape filled with -1
    modified_activations = []
    for arr in activations:
        if arr is None:
            # Create a new array filled with -1 with the valid shape
            new_arr = np.full(valid_shape, -1.0, dtype="float")
            modified_activations.append(new_arr)
        else:
            modified_activations.append(arr)

    # Stack the arrays
    stacked_activations = np.stack(modified_activations)
    return stacked_activations


def push_prototypes(
    dataloader: torch.utils.data.DataLoader,
    prototype_network: PPNet,
    sample_rate: int,
    spectrogram_height: int,
    spectrogram_width: int,
    n_fft: int,
    hop_length: int,
    label_to_category_mapping: Dict[int, str],
    mean: Tuple[float],
    std: Tuple[float],
    n_mels: Optional[int] = None,
    prototype_layer_stride: int = 1,
    root_dir_for_saving_prototypes: Optional[str] = None,
    # save_prototype_class_identity: bool = True,
    save_prototype_waveform_files: bool = False,
    save_prototype_spectrogram_files: bool = False,
) -> None:
    """
    Pushes each prototype to the nearest patch in the training set.

    Args:
    - dataloader (torch.utils.data.DataLoader): DataLoader containing the training or test data.
    - prototype_network (PPNet): Instance of the PPNet as a PyTorch model.
    - sample_rate (int): Sample rate of audio.
    - n_fft (int): FFT window size.
    - hop_length (int): Hop length for FFT.
    - mean (Tuple[float]): Mean for standardization.
    - std (Tuple[float]): Standard deviation for standardization.
    - n_mels (Optional[int]): Number of Mel bands. Default is None.
    - prototype_layer_stride (int): Stride of the prototype layer. Default is 1.
    - root_dir_for_saving_prototypes (Optional[str]): Directory to save the prototypes. Default is None.
    - save_prototype_class_identity (bool): If True, saves the class identity of the prototype spectrograms. Default is
     True.

    Returns:
        None

    Saves the following files (if specified):
    * Prototype spectrograms: `prototype_{prototype_category}_{index}_original_figure.png`
    * Heatmap overlay on spectrograms: `prototype_{prototype_category}_{index}_original_heatmap.png`
    * Bounding box around prototype patch on spectrograms:
    `prototype_{prototype_category}_{index}_original_bounding-box.png`
    * Original audio waveform: `prototype_{prototype_category}_{index}_original_waveform.wav`
    * Prototype audio waveform: `prototype_{prototype_category}_{index}_part_waveform.wav`
    * Additionally, it saves prototype bounding boxes, prototype categories, original prototype spectrograms, and max
    activations in numpy files if a directory is provided.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prototype_network.to(device)

    prototype_network.eval()
    logging.info("push")

    start_total = time.time()

    prototype_shape = prototype_network.prototype_shape
    n_prototypes = prototype_network.num_prototypes

    # Initialize prototype_details dictionary
    prototype_details = {
        "original_spectrograms": np.full(
            shape=[n_prototypes, spectrogram_height, spectrogram_width, 1],
            fill_value=-1,
            dtype=np.float32,
        ),
        "bounding_boxes": np.full(shape=[n_prototypes, 4], fill_value=-1),
        "prototype_labels": np.full(shape=[n_prototypes], fill_value=-1),
        "prototype_categories": np.full(
            shape=[n_prototypes], fill_value="", dtype="U16"
        ),
        "prototype_indices": np.full(shape=[n_prototypes], fill_value=-1),
        "max_prototype_activations": np.full(n_prototypes, -np.inf),
        "prototype_activations": [None] * n_prototypes,
        "max_feature_map_patches": np.zeros(
            [n_prototypes, prototype_shape[1], prototype_shape[2], prototype_shape[3]]
        ),
    }

    # Create directory to save prototypes if specified
    if root_dir_for_saving_prototypes is not None:
        prototype_files_dir = root_dir_for_saving_prototypes
        os.makedirs(prototype_files_dir, exist_ok=True)
    else:
        prototype_files_dir = None

    # search_batch_size = dataloader.batch_size

    num_classes = prototype_network.num_classes

    logging.info("Executing push ...")

    # Iterate through batches and update prototypes
    for push_iter, batch in enumerate(dataloader):
        search_batch_input = batch["input_values"]
        search_y = batch["labels"]

        prototype_details = update_prototypes_on_batch(
            search_batch_unnormalized=search_batch_input,
            prototype_network=prototype_network,
            prototype_details=prototype_details,
            mean=mean,
            std=std,
            search_y=search_y,
            num_classes=num_classes,
            prototype_layer_stride=prototype_layer_stride,
            label_to_category_mapping=label_to_category_mapping,
        )

    prototype_details["prototype_activations"] = stack_prototype_activations(
        prototype_details["prototype_activations"]
    )

    # Save prototype data if a directory is provided
    if prototype_files_dir is not None:
        np.save(
            os.path.join(
                prototype_files_dir,
                "prototype-bounding-boxes.npy",
            ),
            prototype_details["bounding_boxes"],
        )

        np.save(
            os.path.join(
                prototype_files_dir,
                "prototype-categories.npy",
            ),
            prototype_details["prototype_categories"],
        )

        np.save(
            os.path.join(
                prototype_files_dir,
                "prototype-labels.npy",
            ),
            prototype_details["prototype_labels"],
        )

        np.save(
            os.path.join(
                prototype_files_dir,
                "prototype-indices.npy",
            ),
            prototype_details["prototype_indices"],
        )

        # Save the original prototype power spectrograms in decibel units
        np.save(
            os.path.join(
                prototype_files_dir,
                "prototype-spectrograms-original.npy",
            ),
            prototype_details["original_spectrograms"],
        )

        # saves the highest activations
        np.save(
            os.path.join(prototype_files_dir, "prototype-max-activations.npy"),
            prototype_details["max_prototype_activations"],
        )

        # saves the prototype activation maps
        np.save(
            os.path.join(prototype_files_dir, "prototype-activations.npy"),
            prototype_details["prototype_activations"],
        )

        np.save(
            os.path.join(prototype_files_dir, "prototype-max-feature-map-patches.npy"),
            prototype_details["max_feature_map_patches"],
        )

    prototype_update = np.reshape(
        prototype_details["max_feature_map_patches"],
        tuple(prototype_shape),
    )
    prototype_network.prototype_vectors.data.copy_(
        torch.tensor(prototype_update, dtype=torch.float32).to(device)
    )

    if root_dir_for_saving_prototypes is not None and (
        save_prototype_waveform_files or save_prototype_spectrogram_files
    ):
        # Custom handling for LinearLayerWithoutNegativeConnections
        features_per_output_class = (
            prototype_network.last_layer.features_per_output_class
        )

        # Loop over each prototype and update it using the update_prototype function
        for prototype_index in range(n_prototypes):

            prototype_class = prototype_details["prototype_labels"][prototype_index]

            # Calculate the corresponding feature index for the prototype
            prototype_feature_index = prototype_index % features_per_output_class

            # Get the connection weight from the custom layer
            last_layer_connection_prototype_class = prototype_network.last_layer.weight[
                prototype_class, prototype_feature_index
            ]

            if last_layer_connection_prototype_class > 0.0:

                prototype_category = prototype_details["prototype_categories"][
                    prototype_index
                ]
                single_prototype_activations = prototype_details[
                    "prototype_activations"
                ][prototype_index]
                single_prototype_bounding_box = prototype_details["bounding_boxes"][
                    prototype_index
                ]
                original_spectrogram_unnormalized = prototype_details[
                    "original_spectrograms"
                ][prototype_index]

                dir_for_saving_prototypes = os.path.join(
                    root_dir_for_saving_prototypes,
                    str(prototype_category),
                    str(prototype_index),
                )
                os.makedirs(dir_for_saving_prototypes, exist_ok=True)

                # save the numpy array of the prototype self activation
                np.save(
                    os.path.join(
                        dir_for_saving_prototypes,
                        f"prototype-self-activation_{prototype_category}_{prototype_index}.npy",
                    ),
                    single_prototype_activations,
                )

                # create a PyTorch tensor for the original audio spectrogram with shape (N, H, W, C) by adding a
                # batch dimension and a channel dimension
                original_spectrogram_unnormalized_torch = torch.tensor(
                    original_spectrogram_unnormalized
                ).unsqueeze(0)
                # convert shape (N, H, W, C) to shape (N, C, H, W)
                original_spectrogram_unnormalized_torch = torch.permute(
                    original_spectrogram_unnormalized_torch, (0, 3, 1, 2)
                )

                if save_prototype_waveform_files:
                    save_prototype_waveforms(
                        directory=dir_for_saving_prototypes,
                        prototype_category=prototype_category,
                        index=prototype_index,
                        spectrogram=original_spectrogram_unnormalized_torch,
                        bounding_box=single_prototype_bounding_box,
                        sample_rate=sample_rate,
                        n_fft=n_fft,
                        hop_length=hop_length,
                        n_mels=n_mels,
                    )

                if save_prototype_spectrogram_files:
                    # squeeze the channel dimension since we are working with mono channel audio
                    original_spectrogram_unnormalized = (
                        original_spectrogram_unnormalized.squeeze(2)
                    )

                    save_prototype_spectrograms(
                        directory=dir_for_saving_prototypes,
                        prototype_category=prototype_category,
                        index=prototype_index,
                        spectrogram=original_spectrogram_unnormalized,
                        sample_rate=sample_rate,
                        hop_length=hop_length,
                        n_fft=n_fft,
                        n_mels=n_mels,
                        activation_pattern=single_prototype_activations,
                        bounding_box=single_prototype_bounding_box,
                    )

            gc.collect()

    # prototype_network.cuda()
    end_total = time.time()
    logging.info(f"push time: {end_total - start_total}")


def update_prototypes_on_batch(
    search_batch_unnormalized: torch.Tensor,
    label_to_category_mapping: Dict[int, str],
    prototype_network: "LightningProtoPNet",
    prototype_details,
    mean: Tuple[float],
    std: Tuple[float],
    search_y: Optional[Union[torch.Tensor, npt.NDArray[np.float64]]] = None,
    num_classes: Optional[int] = None,
    prototype_layer_stride: int = 1,
) -> Tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[str],
    npt.NDArray[np.float64],
]:
    """
    Updates each prototype of a Prototypical Part Network (PPNet) for the current search batch.

    This function updates the prototypes based on the input batch of spectrograms.
    It processes each prototype, updates the maximum prototype activations, feature map patches,
    and saves relevant prototype data as required.

    Parameters:
    - search_batch_unnormalized (torch.Tensor): Input batch of unnormalized spectrograms for searching prototypes.
    - start_index_of_search_batch (int): Starting index of the search batch.
    - prototype_network (LightningProtoPNet): Instance of the Prototypical Part Network as a PyTorch Lightning model.
    - global_max_prototype_activations (np.ndarray): Array containing the maximum prototype activations across batches.
    - global_max_feature_map_patches (np.ndarray): Array containing the maximum feature map patches across batches.
    - prototype_bounding_boxes (np.ndarray): Array containing the boundaries of the prototypes across batches.
    - prototype_categories (np.ndarray): Array containing the labels (in string format) of the prototypes across
    batches.
    - original_prototype_spectrograms (np.ndarray): Array containing the original spectrograms of the prototypes across
    batches.
    - root_dir_for_saving_prototypes (str): Directory path to save the prototype spectrograms and data.
    - sample_rate (int): Sample rate for audio processing.
    - n_fft (int): FFT size for spectrogram computation.
    - hop_length (int): Hop length for spectrogram computation.
    - mean (Tuple[float]): Mean value for normalization.
    - std (Tuple[float]): Standard deviation value for normalization.
    - n_mels (Optional[int]): Number of Mel bands, if applicable.
    - search_y (Optional[Union[Tensor, np.ndarray]], optional): Integer labels of the input batch.
    - num_classes (Optional[int], optional): Number of classes in the dataset.
    - prototype_layer_stride (int, optional): Stride of the prototype layer. Defaults to 1.

    Raises:
    - ValueError: If search_y or num_classes are not provided.

    Returns:
        None
    """

    # Check if necessary parameters are provided
    if search_y is None or num_classes is None:
        raise ValueError("search_y and num_classes must be provided.")

    # Determine the device to be used for computations (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set the prototype network to evaluation mode
    prototype_network.eval()

    # Normalize the input search batch
    search_batch_normalized = standardize_tensor(
        search_batch_unnormalized, mean=mean, std=std
    )

    with torch.no_grad():
        # Move the normalized search batch to the determined device
        search_batch_normalized = search_batch_normalized.to(device)

        # Compute the prototype activations for the search batch
        (
            protol_input,
            prototype_activations,
        ) = prototype_network.push_forward(search_batch_normalized)

    # Determine if labels are multiclass (integer format) or multilabel (one-hot encoded)
    is_multilabel = len(search_y.shape) == 2 and search_y.shape[1] > 1

    if is_multilabel:
        # Initialize the dictionary mapping class labels to sample indices
        class_to_sample_index_dict = {}

        # Extract the classes present in the batch
        present_classes = (
            (search_y.sum(dim=0) > 0).nonzero(as_tuple=False).flatten().tolist()
        )

        # For multilabel, iterate only through the present classes
        for class_index in present_classes:
            # Find indices where the sample is associated with the current class
            sample_indices = torch.where(search_y[:, class_index] == 1)[0]
            class_to_sample_index_dict[class_index] = sample_indices.tolist()

    else:
        # Initialize the dictionary mapping class labels to sample indices
        class_to_sample_index_dict = {key: [] for key in range(num_classes)}

        for sample_index, sample_label in enumerate(search_y):
            sample_class = sample_label.item()
            class_to_sample_index_dict[sample_class].append(sample_index)

    # # If prototypes are class-specific, create a dictionary mapping class labels to sample indices
    # # Index class_to_sample_index_dict with class number, return list of samples
    # class_to_sample_index_dict = {key: [] for key in range(num_classes)}
    # if is_multilabel:
    #     # For multilabel, iterate through each class and find samples that belong to that class
    #     for class_index in range(num_classes):
    #         # Find indices where the sample is associated with the current class
    #         sample_indices = torch.where(search_y[:, class_index] == 1)[0]
    #         class_to_sample_index_dict[class_index] = sample_indices.tolist()
    # else:
    #     # For multiclass, simply map each sample to its respective class
    #     for sample_index, sample_label in enumerate(search_y):
    #         sample_class = sample_label.item()
    #         class_to_sample_index_dict[sample_class].append(sample_index)

    # Extract prototype shape and total number of prototypes
    prototype_shape = prototype_network.prototype_shape

    # Ensure search_y is on the same device as prototype_class_identity
    search_y = search_y.to(device)

    if is_multilabel:
        batch_size = search_y.shape[0]

        # Expand prototype_class_identity to match the batch size
        expanded_prototype_class_identity = (
            prototype_network.prototype_class_identity.expand(batch_size, -1)
        )

        # Use the class presence mask to index the prototypes of the correct class
        prototypes_of_correct_class = search_y.gather(
            1, expanded_prototype_class_identity.to(device)
        ).float()
    else:
        # Get the prototypes of the correct class for multiclass
        prototypes_of_correct_class = (
            prototype_network.prototype_class_identity.unsqueeze(0).to(device)
            == search_y.unsqueeze(1)
        ).float()

    # Get the indices of the relevant prototypes
    relevant_prototypes_indices = torch.nonzero(
        prototypes_of_correct_class.sum(0)
    ).squeeze(1)

    # Loop over each relevant prototype and update it using the update_prototype function
    for prototype_index in relevant_prototypes_indices:
        prototype_details = update_prototype(
            prototype_index=prototype_index,
            prototype_network=prototype_network,
            prototype_activations=prototype_activations,
            class_to_sample_index_dict=class_to_sample_index_dict,
            prototype_layer_stride=prototype_layer_stride,
            protol_input=protol_input,
            search_batch_unnormalized=search_batch_unnormalized,
            prototype_details=prototype_details,
            prototype_shape=prototype_shape,
            label_to_category_mapping=label_to_category_mapping,
        )

    # Cleanup if class-specific
    del class_to_sample_index_dict

    return prototype_details


def update_prototype(
    prototype_index: int,
    prototype_network: "LightningProtoPNet",
    prototype_activations: npt.NDArray[np.float64],
    class_to_sample_index_dict: dict[int, list[int]],
    prototype_layer_stride: int,
    protol_input: npt.NDArray[np.float64],
    search_batch_unnormalized: torch.Tensor,
    prototype_details,
    prototype_shape: Tuple[int, int, int, int],
    label_to_category_mapping: Dict[int, str],
) -> Tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[str],
    npt.NDArray[np.float64],
]:
    """
    Helper function to update a single prototype during a batch update.

    Parameters:
    - prototype_index: Index of the prototype to be updated.
    - prototype_network: The trained prototype network model.
    - prototype_activations: The activations of the prototypes.
    - original_prototype_spectrograms: The original spectrograms of the prototypes.
    - search_y: Labels of the search batch.
    - class_to_sample_index_dict: Dictionary mapping class indices to their respective sample indices.
    - prototype_layer_stride: Stride of the prototype layer in the network.
    - global_max_prototype_activations: Global maximum activations of the prototypes.
    - global_max_feature_map_patches: Global maximum feature map patches.
    - protol_input: Input to the prototype layer.
    - search_batch_unnormalized: Unnormalized search batch.
    - prototype_categories: Categories of the prototypes.
    - start_index_of_search_batch: Starting index of the search batch.
    - prototype_bounding_boxes: Bounding boxes of the prototypes.
    - root_dir_for_saving_prototypes: Directory for saving prototype data.
    - sample_rate: Sample rate for audio processing.
    - n_fft: FFT size for spectrogram computation.
    - hop_length: Hop length for spectrogram computation.
    - n_mels: Number of Mel bands.
    - prototype_shape: Shape of the prototype.

    Returns:
    - Tuple containing updated values for global_max_prototype_activations, global_max_feature_map_patches,
    prototype_categories,
      and prototype_bounding_boxes.
    """

    # Extract the class index from the prototype index
    class_index = prototype_index

    # If the prototype is class-specific
    # Determine the target class of the prototype
    target_class = prototype_network.prototype_class_identity[class_index].item()

    # If there are no spectrograms of the target class in the batch, return current values
    if target_class not in class_to_sample_index_dict:
        return prototype_details

    # Extract activations specific to the target class
    single_prototype_activations = prototype_activations[
        class_to_sample_index_dict[target_class]
    ][:, prototype_index, :, :]

    single_prototype_activations = single_prototype_activations.cpu().numpy()

    # Determine the maximum activation for the prototype in the batch
    batch_max_single_prototype_activations = np.amax(single_prototype_activations)

    # Check if the batch maximum activation is greater than the global maximum for the prototype
    if (
        batch_max_single_prototype_activations
        # > global_max_prototype_activations[prototype_index]
        > prototype_details["max_prototype_activations"][prototype_index]
    ):
        # If so, update the global maximum and related values

        # Find the location (sample, height, width) of the maximum activation
        batch_argmax_single_prototype_activations = list(
            np.unravel_index(
                np.argmax(single_prototype_activations, axis=None),
                single_prototype_activations.shape,
            )
        )

        # If class-specific, update the index to be relative to the entire batch
        # change the argmin index from the index among spectrograms of the target class to the index in the entire
        # search batch
        batch_argmax_single_prototype_activations[0] = class_to_sample_index_dict[
            target_class
        ][batch_argmax_single_prototype_activations[0]]

        # Extract the corresponding feature map patch based on the maximum activation location
        sample_index_in_batch = batch_argmax_single_prototype_activations[0]
        feature_map_height_start_index = (
            batch_argmax_single_prototype_activations[1] * prototype_layer_stride
        )
        feature_map_width_start_index = (
            batch_argmax_single_prototype_activations[2] * prototype_layer_stride
        )

        # calculate batch_max_feature_map_patch_j
        proto_h = prototype_shape[2]
        proto_w = prototype_shape[3]

        feature_map_height_end_index = feature_map_height_start_index + proto_h
        feature_map_width_end_index = feature_map_width_start_index + proto_w

        # Extract the feature map patch
        batch_max_feature_map_patch_j = protol_input[
            sample_index_in_batch,
            :,
            feature_map_height_start_index:feature_map_height_end_index,
            feature_map_width_start_index:feature_map_width_end_index,
        ]

        # Update the global maximum activation and feature map patch
        prototype_details["max_prototype_activations"][
            prototype_index
        ] = batch_max_single_prototype_activations
        prototype_details["max_feature_map_patches"][
            prototype_index
        ] = batch_max_feature_map_patch_j.cpu().numpy()

        # Process and save the prototype data if required

        # get the whole spectrogram as numpy array
        original_spectrogram_j_unnormalized = search_batch_unnormalized[
            batch_argmax_single_prototype_activations[0]
        ]
        original_spectrogram_j_unnormalized = (
            original_spectrogram_j_unnormalized.numpy()
        )

        # convert shape (C, H, W) to shape (H, W, C)
        original_spectrogram_j_unnormalized = np.transpose(
            original_spectrogram_j_unnormalized, (1, 2, 0)
        )

        original_spectrogram_height = original_spectrogram_j_unnormalized.shape[0]
        original_spectrogram_width = original_spectrogram_j_unnormalized.shape[1]

        # find the highly activated region of the original spectrogram
        single_prototype_activations_sample = prototype_activations[
            sample_index_in_batch, prototype_index, :, :
        ]

        single_prototype_activations_sample = (
            single_prototype_activations_sample.cpu().numpy()
        )

        upsampled_activations = cv2.resize(
            single_prototype_activations_sample,
            dsize=(original_spectrogram_width, original_spectrogram_height),
            interpolation=cv2.INTER_CUBIC,
        )

        single_prototype_bounding_box = find_high_activation_crop(upsampled_activations)

        # Get the label of the prototype (this is actually the label in integer format)
        prototype_label = target_class

        # Get the category of the prototype (this is actually the label in string format)
        prototype_category = label_to_category_mapping[target_class]

        # Update prototype details
        prototype_details["prototype_activations"][
            prototype_index
        ] = single_prototype_activations_sample
        prototype_details["original_spectrograms"][
            prototype_index
        ] = original_spectrogram_j_unnormalized
        prototype_details["bounding_boxes"][
            prototype_index
        ] = single_prototype_bounding_box
        prototype_details["prototype_labels"][prototype_index] = prototype_label
        prototype_details["prototype_categories"][prototype_index] = prototype_category
        prototype_details["prototype_indices"][prototype_index] = prototype_index

    return prototype_details


def save_prototype_spectrograms(
    directory: str,
    prototype_category: str,
    index: int,
    spectrogram: npt.NDArray[np.float64],
    sample_rate: int,
    hop_length: int,
    n_fft: int,
    n_mels: Optional[int],
    activation_pattern: npt.NDArray[np.float64],
    bounding_box: Tuple[int, int, int, int],
) -> None:
    """
    Saves various forms of spectrograms for the prototypes.

    Args:
        directory (str): Directory for saving prototype files.
        prototype_category (str): Prototype category.
        index (int): Index of the prototype.
        spectrogram (npt.NDArray[np.float64]): Spectrogram data.
        sample_rate (int): The sample rate of the audio.
        hop_length (int): The hop length for the STFT calculation.
        n_fft (int): The number of FFT points.
        activation_pattern (npt.NDArray[np.float64]): The activation pattern used to generate the heatmap overlay,
        represented as a 2D numpy array.
        bounding_box (Tuple[int, int, int, int]): Bounding box tuple specifying the prototype patch location.
    """

    base_filename = os.path.join(
        directory, f"prototype_{prototype_category}_{index}_original"
    )

    # Save the entire power spectrogram in decibel units with the prototype as a PNG file
    save_spectrogram_as_figure(
        file_name=f"{base_filename}_figure.pdf",
        spectrogram=spectrogram,
        sample_rate=sample_rate,
        hop_length=hop_length,
        n_fft=n_fft,
        n_mels=n_mels,
    )

    # Save the power spectrogram in decibel units with an overlayed heatmap as a PNG file
    save_spectrogram_with_heatmap(
        file_name=f"{base_filename}_heatmap.pdf",
        spectrogram=spectrogram,
        activation_pattern=activation_pattern,
        sample_rate=sample_rate,
        hop_length=hop_length,
        n_fft=n_fft,
        n_mels=n_mels,
    )

    # Save the whole power spectrogram in decibel units with a bounding box around
    # the prototype patch as a PNG file
    save_spectrogram_with_bounding_box(
        file_name=f"{base_filename}_bounding-box.pdf",
        spectrogram=spectrogram,
        sample_rate=sample_rate,
        hop_length=hop_length,
        n_fft=n_fft,
        n_mels=n_mels,
        bbox_height_start=bounding_box[0],
        bbox_height_end=bounding_box[1],
        bbox_width_start=bounding_box[2],
        bbox_width_end=bounding_box[3],
        color=(21, 101, 249),
    )


def save_prototype_waveforms(
    directory: str,
    prototype_category: str,
    index: int,
    spectrogram: torch.Tensor,
    bounding_box: Tuple[int, int, int, int],
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    n_mels: Optional[int] = None,
) -> None:
    """
    Saves waveforms for the given spectrogram.

    Args:
        directory (str): Directory for saving prototype files.
        prototype_category (str): Prototype category.
        index (int): Index of the prototype.
        spectrogram (torch.Tensor): Spectrogram data represented as a torch tensor.
        bounding_box (Tuple[int, int, int, int]): Bounding box tuple specifying the prototype patch location.
        sample_rate (int): Sample rate of the audio.
        n_fft (int): Number of FFT points.
        hop_length (int): Hop length for the STFT calculation.
        n_mels (Optional[int]): Number of mel filters. Defaults to None.
    """

    base_filename = os.path.join(directory, f"prototype_{prototype_category}_{index}")

    # Save the original audio as waveform
    save_spectrogram_as_waveform(
        spectrogram=spectrogram,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        db_scale=True,
        normalized=False,
        mean=None,
        std=None,
        output_file=f"{base_filename}_original_waveform.wav",
    )

    # Save the prototype as waveform
    save_prototype_as_waveform(
        spectrogram=spectrogram,
        bbox_height_start=bounding_box[0],
        bbox_height_end=bounding_box[1],
        bbox_width_start=bounding_box[2],
        bbox_width_end=bounding_box[3],
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        db_scale=True,
        normalized=False,
        mean=None,
        std=None,
        output_file=f"{base_filename}_part_waveform.wav",
    )
