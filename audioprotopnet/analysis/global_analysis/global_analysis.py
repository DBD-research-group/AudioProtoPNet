import gc
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

from birdset import utils
import cv2
import datasets
import hydra
import lightning as L
import numpy as np
import numpy.typing as npt
from omegaconf import DictConfig
import pyrootutils
from scipy import ndimage
import torch

from audioprotopnet.evaluation.eval_audioprotopnet import initialize_inference_model
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


log = utils.get_pylogger(__name__)
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)

_HYDRA_PARAMS = {
    "version_base": None,
    "config_path": str(root / "configs"),
    "config_name": "main_audioprotopnet.yaml",
}


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


def stack_prototype_activations(
    activations: List[List[Optional[npt.NDArray[np.float64]]]],
) -> npt.NDArray[np.float64]:
    """Stacks a 2D list of prototype activations into a numpy array.

    Args:
        activations (List[List[Optional[npt.NDArray[np.float64]]]]): 2D list of activation arrays or None.

    Returns:
        npt.NDArray[np.float64]: Stacked activation arrays with shape [n_prototypes, top_k, ...].
    """

    # Find the shape of the first non-None array
    valid_shape = None
    for proto_activations in activations:
        for arr in proto_activations:
            if arr is not None:
                valid_shape = arr.shape
                break
        if valid_shape is not None:
            break

    if valid_shape is None:
        raise ValueError(
            "All prototype activations are None, cannot determine a valid shape."
        )

    # Convert None values to arrays of the found valid shape filled with -1
    modified_activations = []
    for proto_activations in activations:
        modified_proto_activations = []
        for arr in proto_activations:
            if arr is None:
                new_arr = np.full(valid_shape, -1.0, dtype=np.float64)
                modified_proto_activations.append(new_arr)
            else:
                modified_proto_activations.append(arr)
        modified_activations.append(modified_proto_activations)

    # Convert to numpy array
    stacked_activations = np.array(modified_activations)
    return stacked_activations


def find_nearest_samples_to_prototypes(
    dataloader: torch.utils.data.DataLoader,
    prototype_network: PPNet,
    sample_rate: int,
    spectrogram_height: int,
    spectrogram_width: int,
    n_fft: int,
    hop_length: int,
    label_to_category_mapping: Dict[int, str],
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
    n_mels: Optional[int] = None,
    prototype_layer_stride: int = 1,
    root_dir_for_saving_prototypes: Optional[str] = None,
    save_prototype_waveform_files: bool = False,
    save_prototype_spectrogram_files: bool = False,
    dataset_labels_to_prototype_labels_mapping: Optional[Dict[int, int]] = None,
    top_k: int = 10,
    class_specific: bool = True,
) -> None:
    """Finds the nearest samples for each prototype in a given dataset.

    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader containing the data.
        prototype_network (PPNet): Instance of the PPNet model.
        sample_rate (int): Sample rate of audio.
        spectrogram_height (int): Height of the spectrograms.
        spectrogram_width (int): Width of the spectrograms.
        n_fft (int): FFT window size.
        hop_length (int): Hop length for FFT.
        label_to_category_mapping (Dict[int, str]): Mapping from labels to category names.
        mean (Tuple[float, float, float]): Mean for standardization.
        std (Tuple[float, float, float]): Standard deviation for standardization.
        n_mels (Optional[int]): Number of Mel bands. Default is None.
        prototype_layer_stride (int): Stride of the prototype layer. Default is 1.
        root_dir_for_saving_prototypes (Optional[str]): Directory to save the prototypes. Default is None.
        save_prototype_waveform_files (bool): If True, saves the prototype waveforms. Default is False.
        save_prototype_spectrogram_files (bool): If True, saves the prototype spectrograms. Default is False.
        dataset_labels_to_prototype_labels_mapping (Optional[Dict[int, int]]): Mapping from dataset labels to prototype labels.
        top_k (int): Number of top activations to consider. Default is 10.

    Returns:
        None
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prototype_network.to(device)

    prototype_network.eval()
    logging.info("")

    start_total = time.time()

    prototype_shape = prototype_network.prototype_shape
    n_prototypes = prototype_network.num_prototypes

    # Predefine prototype labels, categories, and indices
    prototype_labels = (
        prototype_network.prototype_class_identity.cpu().numpy()
    )  # Shape: [n_prototypes]

    # Map prototype labels to categories
    prototype_categories = np.array(
        [label_to_category_mapping[label] for label in prototype_labels]
    )

    # Prototype indices
    prototype_indices = np.arange(n_prototypes)

    # Initialize prototype_details dictionary
    prototype_details = {
        "original_spectrograms": np.full(
            shape=[n_prototypes, top_k, spectrogram_height, spectrogram_width, 1],
            fill_value=-1,
            dtype=np.float32,
        ),
        "bounding_boxes": np.full(shape=[n_prototypes, top_k, 4], fill_value=-1),
        "max_prototype_activations": np.full((n_prototypes, top_k), -np.inf),
        "prototype_activations": [[None] * top_k for _ in range(n_prototypes)],
        "max_feature_map_patches": np.zeros(
            [
                n_prototypes,
                top_k,
                prototype_shape[1],
                prototype_shape[2],
                prototype_shape[3],
            ]
        ),
        "sample_categories": np.full(
            shape=[n_prototypes, top_k], fill_value="", dtype="U256"
        ),
        # Predefined arrays
        "prototype_labels": prototype_labels,  # Shape: [n_prototypes]
        "prototype_categories": prototype_categories,  # Shape: [n_prototypes]
        "prototype_indices": prototype_indices,  # Shape: [n_prototypes]
    }

    # Create directory to save prototypes if specified
    if root_dir_for_saving_prototypes is not None:
        prototype_files_dir = root_dir_for_saving_prototypes
        os.makedirs(prototype_files_dir, exist_ok=True)
    else:
        prototype_files_dir = None

    num_classes = prototype_network.num_classes

    logging.info("Executing finding nearest samples to prototypes ...")

    # Iterate through batches and update prototypes
    for find_iter, batch in enumerate(dataloader):
        search_batch_input = batch["input_values"]
        search_y = batch["labels"]

        # Determine if labels are multiclass (integer format) or multilabel (one-hot encoded)
        is_multilabel = len(search_y.shape) == 2 and search_y.shape[1] > 1

        if is_multilabel:

            # Map the dataset labels to prototype labels
            num_dataset_labels = search_y.shape[1]
            num_prototype_labels = prototype_network.num_classes

            # Create mapping tensor
            mapping_tensor = torch.full((num_dataset_labels,), -1, dtype=torch.long)
            for (
                dataset_label,
                prototype_label,
            ) in dataset_labels_to_prototype_labels_mapping.items():
                mapping_tensor[dataset_label] = prototype_label

            # Identify valid dataset labels
            valid_dataset_labels = (
                (mapping_tensor != -1).nonzero(as_tuple=False).flatten()
            )
            valid_prototype_labels = mapping_tensor[valid_dataset_labels]

            # Extract valid labels
            search_y_valid = search_y[:, valid_dataset_labels]

            # Initialize search_y_mapped
            search_y_mapped = torch.zeros(
                (search_y.shape[0], num_prototype_labels), dtype=search_y.dtype
            )

            # Indices for scatter_add_
            indices = valid_prototype_labels.unsqueeze(0).repeat(search_y.shape[0], 1)

            # Map labels using scatter_add_
            search_y_mapped.scatter_add_(1, indices, search_y_valid)

            # Convert to binary
            search_y_mapped = (search_y_mapped > 0).float()
            search_y_mapped = search_y_mapped.to(device)

        else:
            # Map the dataset labels to prototype labels (Assumes single-label data)
            search_y_mapped = torch.tensor(
                [
                    dataset_labels_to_prototype_labels_mapping[label.item()]
                    for label in search_y
                ]
            ).to(device)

        prototype_details = update_prototypes_on_batch(
            search_batch_unnormalized=search_batch_input,
            prototype_network=prototype_network,
            prototype_details=prototype_details,
            mean=mean,
            std=std,
            search_y=search_y_mapped,
            num_classes=num_classes,
            prototype_layer_stride=prototype_layer_stride,
            top_k=top_k,
            class_specific=class_specific,
            label_to_category_mapping=label_to_category_mapping,
            dataset_labels_to_prototype_labels_mapping=dataset_labels_to_prototype_labels_mapping,
        )

    prototype_details["prototype_activations"] = stack_prototype_activations(
        prototype_details["prototype_activations"]
    )

    # Save prototype data if a directory is provided
    if prototype_files_dir is not None:
        os.makedirs(prototype_files_dir, exist_ok=True)

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

        # save the feature map patches with the highest activations
        np.save(
            os.path.join(prototype_files_dir, "prototype-max-feature-map-patches.npy"),
            prototype_details["max_feature_map_patches"],
        )

        # save the sample categories
        np.save(
            os.path.join(prototype_files_dir, "prototype-sample-categories.npy"),
            prototype_details["sample_categories"],
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

            # Check if the prototype's class corresponds to one of the dataset labels
            if (
                prototype_class
                not in dataset_labels_to_prototype_labels_mapping.values()
            ):
                continue  # Skip this prototype if its label is not in the dataset

            logging.info(
                f"Save files for prototype {prototype_index} with class {prototype_class}"
            )

            prototype_category = prototype_details["prototype_categories"][
                prototype_index
            ]

            # Calculate the corresponding feature index for the prototype
            prototype_feature_index = prototype_index % features_per_output_class

            for i in range(top_k):

                # Check if max_prototype_activations is not -np.inf
                if (
                    prototype_details["max_prototype_activations"][prototype_index][i]
                    == -np.inf
                ):
                    continue  # Skip to the next iteration if it is -np.inf

                # Get the connection weight from the custom layer
                last_layer_connection_prototype_class = (
                    prototype_network.last_layer.weight[
                        prototype_class, prototype_feature_index
                    ]
                )

                if last_layer_connection_prototype_class > 0.0:
                    single_prototype_activations = prototype_details[
                        "prototype_activations"
                    ][prototype_index][i]
                    single_prototype_bounding_box = prototype_details["bounding_boxes"][
                        prototype_index
                    ][i]
                    original_spectrogram_unnormalized = prototype_details[
                        "original_spectrograms"
                    ][prototype_index][i]

                    sample_category = prototype_details["sample_categories"][
                        prototype_index
                    ][i]

                    dir_for_saving_prototypes = os.path.join(
                        root_dir_for_saving_prototypes,
                        str(prototype_category),
                        str(prototype_index),
                        f"top_{i+1}",
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
                            sample_category=sample_category,
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
                            sample_category=sample_category,
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
    logging.info(f"Search time: {end_total - start_total}")


def update_prototypes_on_batch(
    search_batch_unnormalized: torch.Tensor,
    prototype_network: PPNet,
    prototype_details: Dict,
    mean: Tuple[float],
    std: Tuple[float],
    search_y: torch.Tensor,
    num_classes: int,
    label_to_category_mapping: Dict[int, str],
    dataset_labels_to_prototype_labels_mapping: Dict[int, int],
    prototype_layer_stride: int = 1,
    top_k: int = 10,
    class_specific: bool = True,
) -> Dict:
    """
    Updates the prototype details based on the current batch.

    Args:
        search_batch_unnormalized (torch.Tensor): Unnormalized input batch of spectrograms.
        prototype_network (PPNet): The PPNet model instance.
        prototype_details (Dict): Dictionary holding the current prototype details.
        mean (Tuple[float]): Mean used for normalization.
        std (Tuple[float]): Standard deviation used for normalization.
        search_y (torch.Tensor): Labels for the input batch.
        num_classes (int): Number of classes in the dataset.
        prototype_layer_stride (int): Stride of the prototype layer.
        top_k (int): Number of top activations to consider.

    Returns:
        Dict: Updated prototype details.
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
            prototype_layer_input,
            prototype_activations,
        ) = prototype_network.push_forward(search_batch_normalized)

    # Determine if labels are multiclass (integer format) or multilabel (one-hot encoded)
    is_multilabel = len(search_y.shape) == 2 and search_y.shape[1] > 1

    if class_specific:
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

    else:
        # Non-class-specific case: include all samples under a common key
        all_sample_indices = list(range(search_y.size(0)))
        class_to_sample_index_dict = {"all_classes": all_sample_indices}

    # Extract prototype shape and total number of prototypes
    prototype_shape = prototype_network.prototype_shape

    # Ensure search_y is on the same device as prototype_class_identity
    search_y = search_y.to(device)

    if class_specific:
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
    else:
        # Non-class-specific case: only consider prototypes whose class is in the dataset

        # Extract prototype classes present in the dataset
        dataset_prototype_classes = set(
            dataset_labels_to_prototype_labels_mapping.values()
        )

        # Convert to a tensor on the correct device
        dataset_prototype_classes_tensor = torch.tensor(
            list(dataset_prototype_classes), device=device
        )

        # Get the prototype classes from the model
        prototype_classes = prototype_network.prototype_class_identity.to(device)

        # Create a boolean mask for prototypes whose class is in dataset_prototype_classes
        mask = torch.zeros_like(prototype_classes, dtype=torch.bool)
        for c in dataset_prototype_classes_tensor:
            mask |= prototype_classes == c

        # Get indices of relevant prototypes
        relevant_prototypes_indices = torch.nonzero(mask).squeeze(1)

    # Loop over each relevant prototype and update it using the update_prototype function
    for prototype_index in relevant_prototypes_indices:
        prototype_details = update_prototype(
            prototype_index=prototype_index,
            prototype_network=prototype_network,
            prototype_activations=prototype_activations,
            class_to_sample_index_dict=class_to_sample_index_dict,
            prototype_layer_stride=prototype_layer_stride,
            prototype_layer_input=prototype_layer_input,
            search_batch_unnormalized=search_batch_unnormalized,
            prototype_details=prototype_details,
            prototype_shape=prototype_shape,
            top_k=top_k,
            class_specific=class_specific,
            search_y=search_y,
            label_to_category_mapping=label_to_category_mapping,
            is_multilabel=is_multilabel,
        )

    # Cleanup if class-specific
    del class_to_sample_index_dict

    return prototype_details


def update_prototype(
    prototype_index: int,
    prototype_network: PPNet,
    prototype_activations: torch.Tensor,
    class_to_sample_index_dict: Dict[int, List[int]],
    prototype_layer_stride: int,
    prototype_layer_input: torch.Tensor,
    search_batch_unnormalized: torch.Tensor,
    prototype_details: Dict,
    search_y: torch.Tensor,
    label_to_category_mapping: Dict[int, str],
    is_multilabel: bool,
    prototype_shape: Tuple[int, int, int, int],
    top_k: int = 10,
    class_specific: bool = True,
) -> Dict:
    """
    Updates the details of a single prototype.

    Args:
        prototype_index (int): Index of the prototype to update.
        prototype_network (PPNet): The PPNet model instance.
        prototype_activations (torch.Tensor): Activations of the prototypes.
        class_to_sample_index_dict (Dict[int, List[int]]): Mapping from class indices to sample indices.
        prototype_layer_stride (int): Stride of the prototype layer.
        prototype_layer_input (torch.Tensor): Input to the prototype layer.
        search_batch_unnormalized (torch.Tensor): Unnormalized input batch.
        prototype_details (Dict): Current prototype details.
        prototype_shape (Tuple[int, int, int, int]): Shape of the prototype.
        top_k (int): Number of top activations to consider.

    Returns:
        Dict: Updated prototype details.
    """

    if class_specific:
        # Determine the target class of the prototype
        target_class = prototype_network.prototype_class_identity[
            prototype_index
        ].item()

        # If there are no samples of the target class in the batch, return current values
        if target_class not in class_to_sample_index_dict:
            return prototype_details

        # Get indices of samples belonging to the target class
        indices_in_batch = class_to_sample_index_dict[target_class]
        if not indices_in_batch:
            return prototype_details
    else:
        # Non-class-specific case: use all samples
        indices_in_batch = class_to_sample_index_dict["all_classes"]

    # Extract activations specific to the target class and prototype
    single_prototype_activations = prototype_activations[indices_in_batch][
        :, prototype_index, :, :
    ]  # Shape: [num_samples_in_class, h, w]

    # Flatten activation maps per sample
    activations_flat = single_prototype_activations.view(
        single_prototype_activations.size(0), -1
    )  # Shape: [num_samples_in_class, h*w]

    # Compute the maximum activation per sample and corresponding indices
    max_activations_per_sample, _ = activations_flat.max(
        dim=1
    )  # Shape: [num_samples_in_class]

    # Iterate over all samples in the batch
    num_samples = max_activations_per_sample.size(0)
    for idx_in_class in range(num_samples):
        activation_value = max_activations_per_sample[idx_in_class].item()
        sample_index_in_batch = indices_in_batch[idx_in_class]

        # Compare activation_value with current top_k activation values
        current_topk_activations = prototype_details["max_prototype_activations"][
            prototype_index
        ]

        # Determine where to insert this activation value in the top_k list
        insert_pos = None
        for i in range(top_k):
            if activation_value > current_topk_activations[i]:
                insert_pos = i
                break

        if insert_pos is not None:
            # Shift lower activation values down to make room for the new activation
            for k in range(top_k - 1, insert_pos, -1):
                prototype_details["max_prototype_activations"][prototype_index][k] = (
                    prototype_details["max_prototype_activations"][prototype_index][
                        k - 1
                    ]
                )
                prototype_details["max_feature_map_patches"][prototype_index][k] = (
                    prototype_details["max_feature_map_patches"][prototype_index][k - 1]
                )
                prototype_details["original_spectrograms"][prototype_index][k] = (
                    prototype_details["original_spectrograms"][prototype_index][k - 1]
                )
                prototype_details["bounding_boxes"][prototype_index][k] = (
                    prototype_details["bounding_boxes"][prototype_index][k - 1]
                )
                prototype_details["prototype_activations"][prototype_index][k] = (
                    prototype_details["prototype_activations"][prototype_index][k - 1]
                )
                prototype_details["sample_categories"][prototype_index][k] = (
                    prototype_details["sample_categories"][prototype_index][k - 1]
                )

            # Now insert the new activation info at position insert_pos

            # Get the activation map for this sample and prototype
            activation_map = single_prototype_activations[idx_in_class]

            # Find the spatial location of the maximum activation
            max_idx = activations_flat[idx_in_class].argmax().item()
            h = max_idx // activation_map.size(1)
            w = max_idx % activation_map.size(1)

            # Extract feature map patch
            feature_map_height_start_index = h * prototype_layer_stride
            feature_map_width_start_index = w * prototype_layer_stride
            proto_h = prototype_shape[2]
            proto_w = prototype_shape[3]
            feature_map_height_end_index = feature_map_height_start_index + proto_h
            feature_map_width_end_index = feature_map_width_start_index + proto_w
            batch_feature_map_patch = (
                prototype_layer_input[
                    sample_index_in_batch,
                    :,
                    feature_map_height_start_index:feature_map_height_end_index,
                    feature_map_width_start_index:feature_map_width_end_index,
                ]
                .cpu()
                .numpy()
            )

            # Get original spectrogram
            original_spectrogram = search_batch_unnormalized[sample_index_in_batch]
            original_spectrogram = original_spectrogram.cpu().numpy()
            original_spectrogram = np.transpose(original_spectrogram, (1, 2, 0))

            original_spectrogram_height = original_spectrogram.shape[0]
            original_spectrogram_width = original_spectrogram.shape[1]

            # Activation pattern
            activation_pattern = activation_map.cpu().numpy()

            # Bounding box
            upsampled_activations = cv2.resize(
                activation_pattern,
                dsize=(
                    original_spectrogram_width,
                    original_spectrogram_height,
                ),
                interpolation=cv2.INTER_CUBIC,
            )
            bounding_box = find_high_activation_crop(upsampled_activations)

            # Get the sample category
            if is_multilabel:
                # For multilabel, retrieve the actual labels
                sample_labels = (
                    search_y[sample_index_in_batch]
                    .nonzero(as_tuple=False)
                    .flatten()
                    .cpu()
                    .numpy()
                )
                sample_categories = [
                    label_to_category_mapping[label] for label in sample_labels
                ]
                sample_categories.sort()  # Sort alphabetically
                sample_category = "_".join(sample_categories)
            else:
                # For multiclass, the label is a scalar
                label = search_y[sample_index_in_batch].item()
                sample_category = label_to_category_mapping[label]

            # Insert the new activation info at the correct position
            prototype_details["max_prototype_activations"][prototype_index][
                insert_pos
            ] = activation_value
            prototype_details["max_feature_map_patches"][prototype_index][
                insert_pos
            ] = batch_feature_map_patch
            prototype_details["original_spectrograms"][prototype_index][
                insert_pos
            ] = original_spectrogram
            prototype_details["bounding_boxes"][prototype_index][
                insert_pos
            ] = bounding_box
            prototype_details["prototype_activations"][prototype_index][
                insert_pos
            ] = activation_pattern
            prototype_details["sample_categories"][prototype_index][
                insert_pos
            ] = sample_category

    return prototype_details


def save_prototype_spectrograms(
    directory: str,
    prototype_category: str,
    sample_category: str,
    index: int,
    spectrogram: npt.NDArray[np.float64],
    sample_rate: int,
    hop_length: int,
    n_fft: int,
    n_mels: Optional[int],
    activation_pattern: npt.NDArray[np.float64],
    bounding_box: Tuple[int, int, int, int],
) -> None:
    """Saves various forms of spectrograms for the prototypes.

    Args:
        directory (str): Directory for saving prototype files.
        prototype_category (str): Prototype category.
        sample_category (str): Sample category.
        index (int): Index of the prototype.
        spectrogram (npt.NDArray[np.float64]): Spectrogram data (shape: [height, width]).
        sample_rate (int): The sample rate of the audio.
        hop_length (int): The hop length for the STFT calculation.
        n_fft (int): The number of FFT points.
        n_mels (Optional[int]): Number of Mel bands.
        activation_pattern (npt.NDArray[np.float64]): Activation pattern used to generate the heatmap overlay (shape: [height, width]).
        bounding_box (Tuple[int, int, int, int]): Bounding box specifying the prototype patch location (lower_y, upper_y, lower_x, upper_x).

    Returns:
        None
    """

    base_filename = os.path.join(
        directory, f"prototype_{prototype_category}_{index}_sample_{sample_category}"
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
        minimalist=False,
    )

    # Save the whole power spectrogram in decibel units with a bounding box around
    # the prototype patch as a PNG file without axis labels, ticks, titles, or a colorbar.
    save_spectrogram_with_bounding_box(
        file_name=f"{base_filename}_bounding-box_minimalist.pdf",
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
        minimalist=True,
    )


def save_prototype_waveforms(
    directory: str,
    prototype_category: str,
    sample_category: str,
    index: int,
    spectrogram: torch.Tensor,
    bounding_box: Tuple[int, int, int, int],
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    n_mels: Optional[int] = None,
) -> None:
    """Saves waveforms for the given spectrogram.

    Args:
        directory (str): Directory for saving prototype files.
        prototype_category (str): Prototype category.
        sample_category (str): Sample category.
        index (int): Index of the prototype.
        spectrogram (torch.Tensor): Spectrogram data (shape: [1, channels, height, width]).
        bounding_box (Tuple[int, int, int, int]): Bounding box specifying the prototype patch location.
        sample_rate (int): Sample rate of the audio.
        n_fft (int): Number of FFT points.
        hop_length (int): Hop length for the STFT calculation.
        n_mels (Optional[int]): Number of Mel bands.

    Returns:
        None
    """

    base_filename = os.path.join(
        directory, f"prototype_{prototype_category}_{index}_sample_{sample_category}"
    )

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


@utils.register_custom_resolvers(**_HYDRA_PARAMS)
@hydra.main(**_HYDRA_PARAMS)
def global_analysis(cfg: DictConfig):
    """Conducts a comprehensive global analysis on audio data using the configured prototype network.

    This function orchestrates the entire process, including dataset preparation, model loading,
    and detailed analysis of the most activated prototypes for the specified dataset type.

    Args:
        cfg (DictConfig): Configuration dictionary containing paths, parameters, and model settings.

    Returns:
        None
    """

    log.info("Starting Global Analysis")
    log.info(f"Dataset path: <{os.path.abspath(cfg.paths.dataset_path)}>")
    os.makedirs(cfg.paths.dataset_path, exist_ok=True)

    log.info(f"Log path: <{os.path.abspath(cfg.paths.log_dir)}>")
    os.makedirs(cfg.paths.log_dir, exist_ok=True)

    log.info(f"Root Dir:<{os.path.abspath(cfg.paths.log_dir)}>")
    log.info(f"Work Dir:<{os.path.abspath(cfg.paths.work_dir)}>")
    log.info(f"Output Dir:<{os.path.abspath(cfg.paths.output_dir)}>")

    if cfg.get("seed"):
        log.info(f"Seed everything with <{cfg.seed}>")
        L.seed_everything(cfg.seed)

    # Setup data
    log.info(f"Instantiate datamodule <{cfg.datamodule._target_}>")
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    datamodule.prepare_data()  # has to be called before model for len_traindataset!

    ebird_codes_list_prototype_labels = datasets.load_dataset_builder(
        cfg.datamodule.dataset.hf_path, cfg.datamodule.dataset.hf_name
    ).info.features["ebird_code"]
    label_to_category_mapping_prototype_labels = dict(
        enumerate(ebird_codes_list_prototype_labels.names)
    )

    log.info(f"Instantiate push datamodule <{cfg.datamodule_push._target_}>")
    datamodule_push = hydra.utils.instantiate(cfg.datamodule_push)
    datamodule_push.prepare_data()  # has to be called before model for len_traindataset!

    ebird_codes_list_dataset_labels = datasets.load_dataset_builder(
        cfg.datamodule_push.dataset.hf_path, cfg.datamodule_push.dataset.hf_name
    ).info.features["ebird_code"]
    label_to_category_mapping_dataset_labels = dict(
        enumerate(ebird_codes_list_dataset_labels.names)
    )

    # Step 6: Iterate over the train and test dataloaders
    dataset_type = cfg.global_analysis.dataset_type
    if dataset_type == "train":
        # The train dataloader must return normalized power spectrograms in decibel scale without augmentations!
        datamodule_push.setup(stage="fit")
        dataloader = datamodule_push.train_dataloader()
    elif dataset_type == "validation":
        datamodule_push.setup(stage="fit")
        dataloader = datamodule_push.val_dataloader()
    elif dataset_type == "test":
        # The test dataset must return normalized power spectrograms in decibel (dB) scale without augmentations!
        datamodule_push.setup(stage="test")
        dataloader = datamodule_push.test_dataloader()
    else:
        raise NotImplementedError(
            "Currently, only loading files from a train, validation, or test dataset is supported."
        )

    log.info(f"Analyzing {dataset_type} dataset...")

    log.info(f"Instantiate model <{cfg.module.network.model._target_}>")
    # Set up model for the current phase
    model = initialize_inference_model(
        config=cfg,
        train_batch_size=datamodule.loaders_config.train.batch_size,
        len_trainset=datamodule.len_trainset,
        label_counts=datamodule.num_train_labels,
        checkpoint=cfg.ckpt_path,
    )

    prototype_files_dir = os.path.join(cfg.paths.prototype_files_dir, dataset_type)
    os.makedirs(prototype_files_dir, exist_ok=True)

    # Invert the label-to-category mappings
    category_to_prototype_label_mapping = {
        category_name: label
        for label, category_name in label_to_category_mapping_prototype_labels.items()
    }

    category_to_dataset_label_mapping = {
        category_name: label
        for label, category_name in label_to_category_mapping_dataset_labels.items()
    }

    common_categories = set(category_to_prototype_label_mapping.keys()) & set(
        category_to_dataset_label_mapping.keys()
    )

    dataset_labels_to_prototype_labels_mapping = {
        dataset_label: category_to_prototype_label_mapping[category_name]
        for dataset_label, category_name in label_to_category_mapping_dataset_labels.items()
        if category_name in common_categories
    }

    sample_rate = cfg.module.network.sampling_rate
    n_fft = cfg.datamodule.transforms.preprocessing.spectrogram_conversion.n_fft
    hop_length = (
        cfg.datamodule.transforms.preprocessing.spectrogram_conversion.hop_length
    )
    n_mels = cfg.datamodule.transforms.preprocessing.melscale_conversion.n_mels

    save_prototype_spectrogram_files = cfg.save_prototype_spectrogram_files
    save_prototype_waveform_files = cfg.save_prototype_waveform_files

    train_mean = datamodule.transforms.preprocessing.mean
    train_std = datamodule.transforms.preprocessing.std
    prototype_layer_stride = cfg.module.prototype_layer_stride

    # Determine the spectrogram dimensions from the first sample
    first_sample = dataloader.dataset[0]
    spectrogram_height, spectrogram_width = first_sample["input_values"].shape[1:3]

    top_k = cfg.global_analysis.top_k
    class_specific = cfg.global_analysis.class_specific

    # Call the find_nearest_samples_to_prototypes function with necessary parameters
    find_nearest_samples_to_prototypes(
        dataloader=dataloader,
        prototype_network=model.model,
        prototype_layer_stride=prototype_layer_stride,
        root_dir_for_saving_prototypes=prototype_files_dir,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        label_to_category_mapping=label_to_category_mapping_prototype_labels,
        mean=train_mean,
        std=train_std,
        n_mels=n_mels,
        spectrogram_height=spectrogram_height,
        spectrogram_width=spectrogram_width,
        save_prototype_waveform_files=save_prototype_waveform_files,
        save_prototype_spectrogram_files=save_prototype_spectrogram_files,
        dataset_labels_to_prototype_labels_mapping=dataset_labels_to_prototype_labels_mapping,
        top_k=top_k,
        class_specific=class_specific,
    )

    log.info("Global analysis completed successfully.")


if __name__ == "__main__":
    global_analysis()
