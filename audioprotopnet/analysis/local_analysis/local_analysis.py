import json
import logging
import os
from typing import Dict, Optional, Sequence, Tuple, Union

from birdset import utils
import cv2
import datasets
import hydra
import lightning as L
import numpy as np
import torch
from omegaconf import DictConfig
from lightning import LightningModule
import pyrootutils

from audioprotopnet.modules.ppnet.lightningprotopnet import LightningProtoPNet
from audioprotopnet.modules.ppnet.ppnet import LinearLayerWithoutNegativeConnections
from audioprotopnet.modules.ppnet.push import find_high_activation_crop
from audioprotopnet.evaluation.eval_audioprotopnet import initialize_inference_model
from audioprotopnet.preprocessing.normalization import undo_standardize_tensor
from audioprotopnet.saving.copy_files import copy_prototype_files
from audioprotopnet.saving.save_files import save_prototype_files, save_sample_files


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


def get_prototype_max_connections(prototype_network: LightningProtoPNet) -> np.ndarray:
    """
    Get the index of the maximum connection for each prototype.

    Args:
        prototype_network (LightningProtoPNet): Prototype network model.

    Returns:
        np.ndarray: Array of indices of the maximum connection for each prototype.
    """
    # For each prototype, identify the class to which the prototype is most strongly connected.
    prototype_max_connection = torch.argmax(
        prototype_network.model.last_layer.weight, dim=0
    )
    return prototype_max_connection.cpu().numpy()


def check_prototype_identities(
    prototype_files_dir: str,
    prototype_network: LightningProtoPNet,
    logger: logging.Logger,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Check and confirm the identities of prototypes in a prototype network.

    Args:
        prototype_files_dir (str): Directory path where the model and its prototype files are stored.
        prototype_network (LightningProtoPNet): An instance of the prototype network model.
        logger (logging.Logger): Logger instance for logging information and warnings.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Arrays of prototype identities, categories, and maximum connections.
    """
    prototype_labels = np.load(
        os.path.join(prototype_files_dir, "prototype-labels.npy")
    )
    prototype_categories = np.load(
        os.path.join(prototype_files_dir, "prototype-categories.npy")
    )

    logger.info(
        f"Prototypes are chosen from {len(set(prototype_labels))} number of classes."
    )
    logger.info(f"Their class identities are: {prototype_labels}")

    if prototype_network.model.incorrect_class_connection:
        prototype_max_connection = get_prototype_max_connections(prototype_network)
    else:
        prototype_max_connection = prototype_labels

    correct_connections_count = np.sum(prototype_max_connection == prototype_labels)

    if correct_connections_count == prototype_network.model.num_prototypes:
        logger.info("All prototypes connect most strongly to their respective classes.")
    else:
        logger.warning(
            "WARNING: Not all prototypes connect most strongly to their respective classes."
        )

    return prototype_labels, prototype_categories, prototype_max_connection


def prepare_data_for_top_prototype_analysis(
    spectrogram: torch.Tensor, train_mean: float, train_std: float
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Prepares the spectrogram for analysis by normalizing it and transforming its shape.

    This function moves the spectrogram to the GPU if available for efficient computation.
    It applies an undo-standardize operation to get the unnormalized spectrogram. The
    unnormalized spectrogram is then converted to a numpy array and reshaped. The function
    returns the unnormalized spectrogram in both PyTorch tensor and numpy array formats.

    Args:
        spectrogram (torch.Tensor): The input spectrogram tensor.
        train_mean (float): The mean used for normalization in the training dataset.
        train_std (float): The standard deviation used for normalization in the training dataset.

    Returns:
        Tuple[torch.Tensor, np.ndarray]:
            - The unnormalized spectrogram in PyTorch tensor format with original dimensions.
            - The unnormalized spectrogram in numpy array format with reshaped dimensions (H, W).
    """

    # Move the spectrogram to GPU for efficient computation if CUDA is available
    if torch.cuda.is_available():
        spectrogram = spectrogram.cuda()

    # Undo-standardize the spectrogram to get it back to its original scale
    spectrogram_unnormalized_torch = undo_standardize_tensor(
        x=spectrogram, mean=train_mean, std=train_std
    )

    # Convert the spectrogram to a numpy array and reshape it from shape (N, C, H, W) to (H, W)
    spectrogram_unnormalized_np = (
        spectrogram_unnormalized_torch.cpu().numpy().squeeze(0)
    )
    spectrogram_unnormalized_np = np.transpose(spectrogram_unnormalized_np, (1, 2, 0))
    spectrogram_unnormalized_np = spectrogram_unnormalized_np.squeeze(2)

    return spectrogram_unnormalized_torch, spectrogram_unnormalized_np


def forward_pass_through_prototype_network(
    prototype_network: LightningModule,
    spectrogram: torch.Tensor,
    prototype_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Performs a forward pass through the prototype network, capturing various outputs.

    This function processes a normalized spectrogram through the prototype network,
    extracting the logits, classification probabilities, prototype activations, and
    prototype activation patterns for analysis.

    Args:
        prototype_network (LightningModule): The prototype network model.
        spectrogram (torch.Tensor): The normalized spectrogram tensor.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        - The logits output by the network.
        - The classification probabilities.
        - The prototype activations.
        - The prototype activation patterns.
    """
    # Set network to evaluation mode
    prototype_network.eval()

    # Move the network to the same device as the input spectrogram for efficient computation
    device = spectrogram.device
    prototype_network = prototype_network.to(device)

    # Forward pass without gradient computation for analysis
    with torch.no_grad():
        logits, additional_returns = prototype_network(spectrogram)

    if hasattr(prototype_network, "class_mask") and prototype_network.class_mask:
        logits = logits[:, prototype_network.class_mask]

    # Extract classification probabilities from logits
    probabilities = prototype_network.output_activation(logits)

    # Get the prototype activations
    prototype_activations = additional_returns[3]

    # Perform push-forward operation to get conv_output and prototype_activation_patterns
    _, prototype_activation_patterns = prototype_network.model.push_forward(spectrogram)

    if prototype_mask is not None:
        prototype_activations = prototype_activations[:, prototype_mask]
        prototype_activation_patterns = prototype_activation_patterns[
            :, prototype_mask, :, :
        ]

    return logits, probabilities, prototype_activations, prototype_activation_patterns


def analyze_classification_results(
    probabilities: torch.Tensor,
    labels: Tuple[int],
    categories: Tuple[str],
    label_to_category_mapping: Dict[int, str],
    logger: logging.Logger,
    classification_threshold: float = 0.5,
) -> Tuple[int, int, str]:
    """
    Analyzes the classification result of a single item by comparing the predicted label against the actual label and logs the outcome.

    This function is intended to be used when the batch size is 1. It extracts the predicted and actual classes from the logits and labels, respectively,
    determines if the prediction is correct, and logs the results.

    Args:
        probabilities (torch.Tensor): The logits output by the network, representing the predicted class scores. Expected to contain a single item (batch size of 1).
        label (torch.Tensor): The actual label of the data item. Expected to contain a single label (batch size of 1).
        logger (logging.Logger): Logger for outputting information.

    Returns:
        Tuple[int, int, str]: The predicted class, the actual class, and the result of the classification as 'correct' or 'incorrect'.
    """

    # Get the indices of all probabilities that are greater than the threshold
    predicted_labels = tuple(
        torch.where(probabilities[0] > classification_threshold)[0].tolist()
    )

    predicted_categories = tuple(
        label_to_category_mapping[idx] for idx in predicted_labels
    )

    # Determine if the prediction is correct based on the probability threshold
    classification_result = "correct" if predicted_labels == labels else "incorrect"

    # Log the results
    logger.info(
        f"Predicted labels: {predicted_labels} | Actual labels: {labels}, Prediction is {classification_result}."
    )
    logger.info(
        f"Predicted categories: {predicted_categories} | Actual categories: {categories}, Prediction is {classification_result}."
    )

    return predicted_labels, predicted_categories, classification_result


def log_prototype_details(
    prototype_index: int,
    prototype_index_original: int,
    prototype_category: str,
    prototype_labels: Union[Sequence[int], int],
    prototype_labels_original: Union[Sequence[int], int],
    prototype_max_connection: Sequence[int],
    prototype_activations: Union[torch.Tensor, float],
    prototype_network: LightningProtoPNet,
    logger: logging.Logger,
    prototype_rank: Optional[int] = None,
    predicted_labels: Optional[Tuple[int]] = None,
    dataset_labels_to_prototype_labels_mapping: Optional[Dict[int, int]] = None,
) -> None:
    """
    Logs information about a prototype in a prototype network, including its activation details.

    Args:
        prototype_index (int): The index of the prototype in the network.
        prototype_category (str): The category of the prototype.
        prototype_labels (Union[Sequence[int], int]): The labels of the prototypes in integer format.
        prototype_max_connection (Sequence[int]): The max connection values of the prototypes.
        prototype_activations (Union[torch.Tensor, float]): The activation values of the prototypes.
        prototype_network (LightningProtoPNet): The prototype network model.
        logger (logging.Logger): Logger for logging information.
        class_label (int): The class index associated with the prototype.
        prototype_rank (Optional[int]): The rank of the prototype among the most activated (if applicable).

    Returns:
        None: This function only logs information and does not return any value.
    """

    # Check if prototype_labels is a numpy.ndarray
    if isinstance(prototype_labels, np.ndarray):
        prototype_label = prototype_labels[prototype_index]
        prototype_label_original = prototype_labels_original[prototype_index]
    # Check if prototype_labels is a numpy.int64
    elif isinstance(prototype_labels, np.int64):
        prototype_label = int(prototype_labels)
        prototype_label_original = int(prototype_labels_original)
    # Raise an error if prototype_labels is neither a numpy.ndarray nor a numpy.int64
    else:
        raise TypeError(
            f"prototype_labels must be either a numpy.ndarray or a numpy.int64, not {type(prototype_labels)}"
        )

    # Check if prototype_activations is a 0-dim tensor or not
    if prototype_activations.ndim == 0:
        activation_value = prototype_activations
    else:
        activation_value = prototype_activations[0][prototype_index]

    if prototype_rank is not None:
        logger.info(f"Top {prototype_rank} activated prototype for this sample:")

    logger.info(f"Prototype index: {prototype_index_original}")
    logger.info(f"Prototype class category: {prototype_category}")
    logger.info(f"Prototype class label: {prototype_label}")

    logger.info(
        f"Prototype max connection label: {prototype_max_connection[prototype_index]}"
    )

    logger.info(f"Activation value (similarity score): {activation_value}")

    # Check if the last layer is an instance of LinearLayerWithoutNegativeConnections
    if isinstance(
        prototype_network.model.last_layer, LinearLayerWithoutNegativeConnections
    ):
        # Custom handling for LinearLayerWithoutNegativeConnections
        features_per_output_class = (
            prototype_network.model.last_layer.features_per_output_class
        )

        # Calculate the corresponding feature index for the prototype
        prototype_feature_index = prototype_index_original % features_per_output_class

        # Get the connection weight from the custom layer
        last_layer_connection_prototype_label = (
            prototype_network.model.last_layer.weight[
                prototype_label_original, prototype_feature_index
            ]
        )
    else:
        # Conventional linear layer handling
        last_layer_connection_prototype_label = (
            prototype_network.model.last_layer.weight[prototype_label_original][
                prototype_index_original
            ]
        )

    logger.info(
        f"Last layer connection to prototype label: {last_layer_connection_prototype_label}"
    )

    if predicted_labels and prototype_network.model.incorrect_class_connection:
        for predicted_label in predicted_labels:
            predicted_label_original = dataset_labels_to_prototype_labels_mapping[
                predicted_label
            ]

            last_layer_connection_predicted_label = (
                prototype_network.model.last_layer.weight[predicted_label_original][
                    prototype_index_original
                ]
            )
            logger.info(
                f"Last layer connection to predicted label {predicted_label}: {last_layer_connection_predicted_label}"
            )


def update_classification_information(
    classification_information: dict,
    prototype_index: int,
    prototype_index_original: int,
    prototype_category: str,
    prototype_labels: Sequence[int],
    prototype_labels_original: Sequence[int],
    prototype_max_connection: Sequence[int],
    prototype_activations: torch.Tensor,
    prototype_network: LightningModule,
    label_to_category_mapping: Dict[int, str],
    prototype_rank: Optional[int] = None,
    class_index: Optional[int] = None,
    prototype_count: Optional[int] = None,
    predicted_labels: Optional[Tuple[int]] = None,
    dataset_labels_to_prototype_labels_mapping: Optional[Dict[int, int]] = None,
) -> dict:
    """
    Updates the classification information dictionary with details about a prototype.

    This function can handle both cases: updating information for the most activated prototypes
    or for the top class prototypes, depending on the provided parameters.

    Args:
        classification_information (dict): The dictionary to be updated.
        prototype_index (int): The index of the prototype in the network.
        prototype_category (str): The category of the prototype.
        prototype_labels (Sequence[int]): The labels of the prototypes.
        prototype_max_connection (Sequence[int]): The max connection values of the prototypes.
        prototype_activations (torch.Tensor): The activation values of the prototypes.
        prototype_network (LightningProtoPNet): The prototype network model.
        class_label (int): The class label.
        label_to_category_mapping (Dict[str, str]): Mapping from label indices to category names.
        prototype_rank (Optional[int], optional): The rank of the prototype among the most activated.
        class_index (Optional[int], optional): The index of the top predicted class.
        prototype_count (Optional[int], optional): The count of the prototype within its class.

    Returns:
        dict: The updated classification information dictionary.
    """
    # Check if prototype_labels is a numpy.ndarray
    if isinstance(prototype_labels, np.ndarray):
        prototype_label = prototype_labels[prototype_index]
        prototype_label_original = prototype_labels_original[prototype_index]
    # Check if prototype_labels is a numpy.int64
    elif isinstance(prototype_labels, np.int64):
        prototype_label = int(prototype_labels)
        prototype_label_original = int(prototype_labels_original)
    # Raise an error if prototype_labels is neither a numpy.ndarray nor a numpy.int64
    else:
        raise TypeError(
            f"prototype_labels must be either a numpy.ndarray or a numpy.int64, not {type(prototype_labels)}"
        )

    prototype_connection_category = label_to_category_mapping[
        prototype_max_connection[prototype_index]
    ]

    # Check if the last layer is an instance of LinearLayerWithoutNegativeConnections
    if isinstance(
        prototype_network.model.last_layer, LinearLayerWithoutNegativeConnections
    ):
        # Custom handling for LinearLayerWithoutNegativeConnections
        features_per_output_class = (
            prototype_network.model.last_layer.features_per_output_class
        )

        # Calculate the corresponding feature index for the prototype
        prototype_feature_index = prototype_index_original % features_per_output_class

        # Get the connection weight from the custom layer
        last_layer_connection_prototype_label = (
            prototype_network.model.last_layer.weight[
                prototype_label_original, prototype_feature_index
            ].tolist()
        )
    else:
        # Conventional linear layer handling
        last_layer_connection_prototype_label = (
            prototype_network.model.last_layer.weight[prototype_label_original][
                prototype_index_original
            ].tolist()
        )

    # Check if prototype_activations is a torch.Tensor or a float
    if isinstance(prototype_activations, torch.Tensor):
        activation_value = prototype_activations[0][prototype_index].tolist()
    else:
        activation_value = prototype_activations

    if prototype_rank is not None:
        key = f"top_{prototype_rank}_activated_prototype"
    else:
        prototype_info_key = f"top_{class_index + 1}_class_prototypes"
        prototype_key = f"prototype_{prototype_count}"
        classification_information.setdefault(prototype_info_key, {})
        key = (prototype_info_key, prototype_key)

    prototype_info = {
        "prototype_index": int(prototype_index_original),
        "prototype_class_category": str(prototype_category),
        "prototype_class_label": int(prototype_label),
        "prototype_connection_category": str(prototype_connection_category),
        "prototype_connection_label": int(prototype_max_connection[prototype_index]),
        "activation_value_(similarity_score)": activation_value,
        "last_layer_connection_prototype_label": last_layer_connection_prototype_label,
    }

    if predicted_labels and prototype_network.model.incorrect_class_connection:
        prototype_info["last_layer_connection_predicted_labels"] = {
            predicted_label: prototype_network.model.last_layer.weight[
                dataset_labels_to_prototype_labels_mapping[predicted_label]
            ][prototype_index_original].tolist()
            for predicted_label in predicted_labels
        }

    if isinstance(key, tuple):
        classification_information[key[0]][key[1]] = prototype_info
    else:
        classification_information[key] = prototype_info

    return classification_information


def analyze_and_upsample_activation_pattern(
    prototype_activation_patterns: torch.Tensor,
    prototype_index: int,
    spectrogram_width: int,
    spectrogram_height: int,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Analyzes and upsamples the activation pattern of a specific prototype.

    This function takes the activation pattern of a prototype, upsamples it to match the dimensions
    of the spectrogram, and identifies the most highly activated patch.

    Args:
        prototype_activation_patterns (torch.Tensor): The activation patterns of all prototypes.
        prototype_index (int): The index of the specific prototype to analyze.
        spectrogram_width (int): The width of the spectrogram to which the pattern will be upsampled.
        spectrogram_height (int): The height of the spectrogram to which the pattern will be upsampled.

    Returns:
        Tuple[np.ndarray, Tuple[int, int, int, int]]: The activation pattern and the bounding box coordinates of the highly activated patch.
    """
    # Extract the activation pattern for the specific prototype
    activation_pattern = (
        prototype_activation_patterns[0][prototype_index].detach().cpu().numpy()
    )

    # Upsample the activation pattern
    # Upsampling is necessary to match the size of the spectrogram for accurate localization of the activated patch
    upsampled_activation_pattern = cv2.resize(
        activation_pattern,
        dsize=(spectrogram_width, spectrogram_height),
        interpolation=cv2.INTER_CUBIC,
    )

    # Find the most highly activated patch of the spectrogram by this prototype
    # This helps in localizing the area in the spectrogram where the prototype is most active
    bounding_box_high_activation_patch = find_high_activation_crop(
        upsampled_activation_pattern
    )

    return activation_pattern, bounding_box_high_activation_patch


def analyze_most_activated_prototypes(
    prototype_network: LightningModule,
    label_to_category_mapping: Dict[int, str],
    prototype_labels: Sequence[int],
    prototype_labels_original: Sequence[int],
    prototype_categories: Sequence[str],
    prototype_max_connection: Sequence[int],
    prototype_indices_original: Sequence[int],
    train_mean: float,
    train_std: float,
    spectrogram_width: int,
    spectrogram_height: int,
    spectrogram: torch.Tensor,
    labels: Tuple[int],
    categories: Tuple[str],
    config: DictConfig,
    logger: logging.Logger,
    local_analysis_dir: str,
    prototype_files_dir: str,
    sample_rate: int,
    hop_length: int,
    n_fft: int,
    n_mels: Optional[int],
    classification_threshold: float,
    save_prototype_spectrogram_files: bool,
    save_prototype_waveform_files: bool,
    dataset_labels_to_prototype_labels_mapping: Dict[int, int],
    prototype_mask: Optional[torch.Tensor] = None,
) -> Tuple[
    str,
    torch.Tensor,
    np.ndarray,
    torch.Tensor,
    torch.Tensor,
    Dict[str, any],
    torch.Tensor,
    np.ndarray,
]:
    """
    Analyzes the most activated prototypes of a spectrogram and compiles classification information.

    Args:
        prototype_network (LightningProtoPNet): The prototype network model.
        label_to_category_mapping (Dict[str, str]): Mapping from label indices to category names.
        prototype_labels (Sequence[int]): The identities of the prototypes.
        prototype_categories (Sequence[str]): The categories of the prototypes.
        prototype_max_connection (Sequence[int]): The max connection values of the prototypes.
        train_mean (float): The mean used for normalization in the training dataset.
        train_std (float): The standard deviation used for normalization in the training dataset.
        dataset_type (str): The type of dataset being analyzed.
        spectrogram_width (int): The width of the spectrogram.
        spectrogram_height (int): The height of the spectrogram.
        sample_index (int): The index of the current sample.
        spectrogram (torch.Tensor): The spectrogram tensor.
        label (int): The label of the sample.
        category (str): The category of the sample.
        config (DictConfig): Configuration object containing parameters.
        logger (logging.Logger): Logger for logging information.

    Returns:
        Tuple[str, torch.Tensor, np.ndarray, torch.Tensor, torch.Tensor, Dict[str, any], torch.Tensor, np.ndarray]:
        Tuple containing directory path, unnormalized spectrogram tensors, logits, probabilities,
        classification information, prototype activations, and prototype activation patterns.
    """

    logger.info("Start analyzing the most activated prototypes.")

    # Forward pass through the prototype network
    (
        logits,
        probabilities,
        prototype_activations,
        prototype_activation_patterns,
    ) = forward_pass_through_prototype_network(
        prototype_network=prototype_network,
        spectrogram=spectrogram,
        prototype_mask=prototype_mask,
    )

    # Prepare spectrogram data for analysis
    (
        spectrogram_unnormalized_torch,
        spectrogram_unnormalized_np_without_channel_dim,
    ) = prepare_data_for_top_prototype_analysis(
        spectrogram=spectrogram, train_mean=train_mean, train_std=train_std
    )

    # Analyze classification results
    (
        predicted_labels,
        predicted_categories,
        classification_result,
    ) = analyze_classification_results(
        probabilities=probabilities,
        labels=labels,
        categories=categories,
        label_to_category_mapping=label_to_category_mapping,
        logger=logger,
        classification_threshold=classification_threshold,
    )

    logger.info(f"path: {local_analysis_dir}")
    os.makedirs(local_analysis_dir, exist_ok=True)

    # Create a directory for most activated prototypes
    os.makedirs(
        os.path.join(local_analysis_dir, "most-activated-prototypes"),
        exist_ok=True,
    )

    # Initialize classification information dictionary
    # This dictionary contains information about the classification of the sample.
    classification_information = {
        "categories": categories,
        "labels": labels,
        "predicted_categories": predicted_categories,
        "predicted_labels": predicted_labels,
        "logits": logits[0].tolist(),
        "probabilities": probabilities[0].tolist(),
        "classification_result": classification_result,
    }

    # Save sample files
    save_sample_files(
        spectrogram_np=spectrogram_unnormalized_np_without_channel_dim,
        spectrogram_torch=spectrogram_unnormalized_torch,
        local_analysis_dir=local_analysis_dir,
        sample_rate=sample_rate,
        hop_length=hop_length,
        n_fft=n_fft,
        n_mels=n_mels,
        save_prototype_waveform_files=save_prototype_waveform_files,
        save_prototype_spectrogram_files=save_prototype_spectrogram_files,
    )

    # Analysis of most activated prototypes
    logger.info(
        f"Most activated {config.local_analysis.number_most_activated_prototypes} prototypes of this sample:"
    )

    # Sort prototype activations and analyze each top activated prototype
    array_act, sorted_indices_act = torch.sort(prototype_activations[0])
    for prototype_rank in range(
        1, config.local_analysis.number_most_activated_prototypes + 1
    ):
        prototype_index = sorted_indices_act[-prototype_rank].item()

        prototype_index_original = prototype_indices_original[prototype_index]
        prototype_category = prototype_categories[prototype_index]
        prototype_activation_value = array_act[-prototype_rank]
        prototype_activation_value_list = prototype_activation_value.tolist()

        # Log prototype details
        log_prototype_details(
            prototype_index=prototype_index,
            prototype_index_original=prototype_index_original,
            prototype_category=prototype_category,
            prototype_labels=prototype_labels,
            prototype_labels_original=prototype_labels_original,
            prototype_max_connection=prototype_max_connection,
            prototype_activations=prototype_activation_value,
            prototype_network=prototype_network,
            logger=logger,
            prototype_rank=prototype_rank,
            predicted_labels=predicted_labels,
            dataset_labels_to_prototype_labels_mapping=dataset_labels_to_prototype_labels_mapping,
        )

        # Copy files related to the prototype
        copy_prototype_files(
            prototype_index_original=prototype_index_original,
            prototype_category=prototype_category,
            local_analysis_dir=local_analysis_dir,
            prototype_files_dir=prototype_files_dir,
            prototype_rank=prototype_rank,
            class_index=None,
            prototype_count=None,
            logger=logger,
        )

        # copy_prototype_files(
        #     prototype_index_original=prototype_index_original,
        #     prototype_category=prototype_category,
        #     local_analysis_dir=local_analysis_dir,
        #     prototype_files_dir=prototype_files_dir,
        #     prototype_rank=prototype_rank,
        #     class_index=None,
        #     prototype_count=None,
        #     logger=logger,
        #     save_prototype_spectrogram_files=save_prototype_spectrogram_files,
        #     save_prototype_waveform_files=save_prototype_waveform_files,
        # )

        # Analyze and upsample the activation pattern
        (
            activation_pattern,
            bounding_box_high_activation_patch,
        ) = analyze_and_upsample_activation_pattern(
            prototype_activation_patterns=prototype_activation_patterns,
            prototype_index=prototype_index,
            spectrogram_width=spectrogram_width,
            spectrogram_height=spectrogram_height,
        )

        # Update classification information with prototype details
        classification_information = update_classification_information(
            classification_information=classification_information,
            prototype_index=prototype_index,
            prototype_index_original=prototype_index_original,
            prototype_category=prototype_category,
            prototype_labels=prototype_labels,
            prototype_labels_original=prototype_labels_original,
            prototype_max_connection=prototype_max_connection,
            prototype_activations=prototype_activation_value_list,
            prototype_network=prototype_network,
            label_to_category_mapping=label_to_category_mapping,
            prototype_rank=prototype_rank,
            class_index=None,
            prototype_count=None,
            predicted_labels=predicted_labels,
            dataset_labels_to_prototype_labels_mapping=dataset_labels_to_prototype_labels_mapping,
        )

        # Save prototype files
        save_prototype_files(
            spectrogram_unnormalized_torch=spectrogram_unnormalized_torch,
            spectrogram_unnormalized_np=spectrogram_unnormalized_np_without_channel_dim,
            bounding_box_high_activation_patch=bounding_box_high_activation_patch,
            activation_pattern=activation_pattern,
            local_analysis_dir=local_analysis_dir,
            logger=logger,
            prototype_rank=prototype_rank,
            class_index=None,
            prototype_count=None,
            sample_rate=sample_rate,
            hop_length=hop_length,
            n_fft=n_fft,
            n_mels=n_mels,
            save_prototype_spectrogram_files=save_prototype_spectrogram_files,
            save_prototype_waveform_files=save_prototype_waveform_files,
        )

    logger.info("Finished analyzing the most activated prototypes.")

    # Return collected data
    return (
        spectrogram_unnormalized_torch,
        spectrogram_unnormalized_np_without_channel_dim,
        logits,
        probabilities,
        classification_information,
        prototype_activations,
        prototype_activation_patterns,
    )


def analyze_top_classes(
    prototype_network: LightningProtoPNet,
    local_analysis_dir: str,
    prototype_labels: Sequence[int],
    prototype_labels_original: Sequence[int],
    prototype_categories: Sequence[str],
    prototype_activations: torch.Tensor,
    prototype_activation_patterns: torch.Tensor,
    prototype_max_connection: Sequence[int],
    prototype_indices_original: Sequence[int],
    spectrogram_height: int,
    spectrogram_width: int,
    label_to_category_mapping: Dict[int, str],
    logits: torch.Tensor,
    probabilities: torch.Tensor,
    classification_information: Dict[str, any],
    spectrogram_unnormalized_torch: torch.Tensor,
    spectrogram_unnormalized_np_without_channel_dim: np.ndarray,
    logger: logging.Logger,
    prototype_files_dir: str,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    n_mels: Optional[int],
    num_top_classes: int,
    save_prototype_spectrogram_files: bool,
    save_prototype_waveform_files: bool,
) -> None:
    """
    Analyzes prototypes from the top predicted classes of a sample, saves various visualizations,
    and updates classification information.

    Args:
        prototype_network (LightningProtoPNet): The prototype network model for analysis.
        local_analysis_dir (str): Directory where the analysis results will be saved.
        prototype_labels (Sequence[int]): Identities of the prototypes.
        prototype_categories (Sequence[str]): Categories of the prototypes.
        prototype_activations (torch.Tensor): Activation values of the prototypes.
        prototype_activation_patterns (torch.Tensor): Activation patterns of the prototypes.
        prototype_max_connection (Sequence[int]): Max connection values of the prototypes.
        spectrogram_height (int): Height of the spectrogram.
        spectrogram_width (int): Width of the spectrogram.
        label_to_category_mapping (Dict[str, str]): Mapping from label indices to category names.
        logits (torch.Tensor): Logits from the network prediction.
        probabilities (torch.Tensor): Probabilities computed from logits.
        classification_information (Dict[str, any]): Dictionary for storing classification info.
        sample_index (int): Index of the current sample being analyzed.
        spectrogram_unnormalized_torch (torch.Tensor): Unnormalized spectrogram tensor.
        spectrogram_unnormalized_np_without_channel_dim (np.ndarray): Unnormalized spectrogram numpy array.
        config (DictConfig): Configuration object containing various parameters.
        logger (logging.Logger): Logger for information logging.

    Returns:
        None: This function performs analysis and file operations without returning any value.
    """
    logger.info(f"Prototypes from top-{num_top_classes} classes:")

    # Extract top-k classes based on logits and probabilities
    topk_logits, topk_classes = torch.topk(logits[0], k=num_top_classes)
    topk_probabilities, _ = torch.topk(probabilities[0], k=num_top_classes)

    # Iterate through each top predicted class
    for class_index, class_label in enumerate(topk_classes.detach().cpu().numpy()):
        # Create directory for prototypes of each top class
        os.makedirs(
            os.path.join(local_analysis_dir, f"top-{class_index+1}-class-prototypes"),
            exist_ok=True,
        )

        class_category = label_to_category_mapping[class_label]

        # Update classification information for the current class
        classification_information[f"top_{class_index + 1}_class_prototypes"] = {
            "class_label": int(class_label),
            "class_category": class_category,
            "logit": topk_logits[class_index].tolist(),
            "probability": topk_probabilities[class_index].tolist(),
        }

        logger.info(
            f"top {class_index+1} predicted class - Label: {class_label} | Category: {class_category}"
        )
        logger.info(f"logit of the class: {topk_logits[class_index]}")

        # Identify prototypes associated with the current class
        class_prototype_indices = np.where(prototype_labels == class_label)[0]

        # class_prototype_indices = np.nonzero(
        #     prototype_network.model.prototype_class_identity.detach()
        #     .cpu()
        #     .numpy()[:, class_label_original]
        # )[0]

        class_prototype_activations = prototype_activations[0][class_prototype_indices]
        _, sorted_indices_cls_act = torch.sort(class_prototype_activations)

        prototype_count = 1

        # Iterate over sorted prototypes for the current class
        for j in reversed(sorted_indices_cls_act.detach().cpu().numpy()):
            prototype_index = class_prototype_indices[j]
            prototype_index_original = prototype_indices_original[prototype_index]
            prototype_category = prototype_categories[prototype_index]
            prototype_label = prototype_labels[prototype_index]
            prototype_label_original = prototype_labels_original[prototype_index]

            # Update classification information with prototype details
            classification_information = update_classification_information(
                classification_information=classification_information,
                prototype_index=prototype_index,
                prototype_index_original=prototype_index_original,
                prototype_category=prototype_category,
                prototype_labels=prototype_label,
                prototype_labels_original=prototype_label_original,
                prototype_max_connection=prototype_max_connection,
                prototype_activations=prototype_activations,
                prototype_network=prototype_network,
                label_to_category_mapping=label_to_category_mapping,
                prototype_rank=None,
                class_index=class_index,
                prototype_count=prototype_count,
                predicted_labels=None,
                dataset_labels_to_prototype_labels_mapping=None,
            )

            # Log details related to the prototype
            log_prototype_details(
                prototype_index=prototype_index,
                prototype_index_original=prototype_index_original,
                prototype_category=prototype_category,
                prototype_labels=prototype_label,
                prototype_labels_original=prototype_label_original,
                prototype_max_connection=prototype_max_connection,
                prototype_activations=prototype_activations,
                prototype_network=prototype_network,
                logger=logger,
                prototype_rank=None,
                predicted_labels=None,
                dataset_labels_to_prototype_labels_mapping=None,
            )

            prototype_index_original = prototype_indices_original[prototype_index]

            # Copy files related to the prototype
            copy_prototype_files(
                prototype_index_original=prototype_index_original,
                prototype_category=prototype_category,
                local_analysis_dir=local_analysis_dir,
                prototype_files_dir=prototype_files_dir,
                prototype_rank=None,
                class_index=class_index,
                prototype_count=prototype_count,
                logger=logger,
            )

            # copy_prototype_files(
            #     prototype_index_original=prototype_index_original,
            #     prototype_category=prototype_category,
            #     local_analysis_dir=local_analysis_dir,
            #     prototype_files_dir=prototype_files_dir,
            #     prototype_rank=None,
            #     class_index=class_index,
            #     prototype_count=prototype_count,
            #     logger=logger,
            #     save_prototype_waveform_files=save_prototype_waveform_files,
            #     save_prototype_spectrogram_files=save_prototype_spectrogram_files,
            # )

            # Analyze and upsample the activation pattern
            (
                activation_pattern,
                bounding_box_high_activation_patch,
            ) = analyze_and_upsample_activation_pattern(
                prototype_activation_patterns=prototype_activation_patterns,
                prototype_index=prototype_index,
                spectrogram_width=spectrogram_width,
                spectrogram_height=spectrogram_height,
            )

            # Save prototype visualization files
            save_prototype_files(
                spectrogram_unnormalized_torch=spectrogram_unnormalized_torch,
                spectrogram_unnormalized_np=spectrogram_unnormalized_np_without_channel_dim,
                bounding_box_high_activation_patch=bounding_box_high_activation_patch,
                activation_pattern=activation_pattern,
                local_analysis_dir=local_analysis_dir,
                logger=logger,
                prototype_rank=None,
                class_index=class_index,
                prototype_count=prototype_count,
                sample_rate=sample_rate,
                hop_length=hop_length,
                n_fft=n_fft,
                n_mels=n_mels,
                save_prototype_spectrogram_files=save_prototype_spectrogram_files,
                save_prototype_waveform_files=save_prototype_waveform_files,
            )

            logger.info(
                "--------------------------------------------------------------"
            )
            prototype_count += 1
        logger.info("***************************************************************")

    # Save classification information as JSON
    with open(
        os.path.join(local_analysis_dir, "classification-information.json"), "w"
    ) as json_file:
        json.dump(classification_information, json_file)


@utils.register_custom_resolvers(**_HYDRA_PARAMS)
@hydra.main(**_HYDRA_PARAMS)
def local_analysis(cfg: DictConfig):
    """
    Conducts a comprehensive local analysis on audio data using the configured prototype network.
    This function orchestrates the entire process, including dataset preparation, model loading and validation,
    prototype identity checking, and detailed analysis of the most activated prototypes for the training, validation,
    and testing datasets.

    The analysis involves:
    - Preparing datasets and dataloaders.
    - Loading and validating the prototype network model.
    - Checking the identities of the prototypes in the network.
    - Analyzing the most activated prototypes for each sample in the datasets.
    - Analyzing the prototypes from the top predicted classes for each sample.

    The function saves various visualizations and logs detailed information about the analysis process.

    Args:
        config (DictConfig): Configuration dictionary containing paths, parameters, and model settings.

    Returns:
        None: This function performs analysis, file operations, and logging, without returning any value.
    """

    log.info("Starting Local Analysis")
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

    # Set the augmentations in the config file to None, since we do not want augmentations for the mean and
    # standard deviation calculations.
    datamodule.transforms.waveform_augmentations = []
    datamodule.transforms.spectrogram_augmentations = []
    datamodule.loaders_config.train.batch_size = 1
    datamodule.loaders_config.valid.batch_size = 1
    datamodule.loaders_config.test.batch_size = 1
    datamodule.loaders_config.train.shuffle = False
    datamodule.loaders_config.valid.shuffle = False
    datamodule.loaders_config.test.shuffle = False

    datamodule.prepare_data()  # has to be called before model for len_traindataset!

    if cfg.datamodule.dataset.dataset_name == "esc50":
        label_to_category_mapping = datamodule.label_to_category_mapping
    else:
        ebird_codes_list = datasets.load_dataset_builder(
            cfg.datamodule.dataset.hf_path, cfg.datamodule.dataset.hf_name
        ).info.features["ebird_code"]
        label_to_category_mapping = dict(enumerate(ebird_codes_list.names))

    # Step 6: Iterate over the train and test dataloaders
    dataset_type = cfg.local_analysis.dataset_type
    if dataset_type == "train":
        # The train dataloader must return normalized power spectrograms in decibel scale without augmentations!
        datamodule.setup(stage="fit")
        dataloader = datamodule.train_dataloader()
    elif dataset_type == "validation":
        datamodule.setup(stage="fit")
        dataloader = datamodule.val_dataloader()
    elif dataset_type == "test":
        # The test dataset must return normalized power spectrograms in decibel (dB) scale without augmentations!
        datamodule.setup(stage="test")
        dataloader = datamodule.test_dataloader()
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

    log.info("Instantiate logger")
    logger = utils.instantiate_loggers(cfg.get("logger"))

    trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    log.info("Logging Hyperparams")
    utils.log_hyperparameters(object_dict)

    local_analysis_dir = cfg.paths.local_analysis_dir
    prototype_files_dir = cfg.paths.prototype_files_dir
    os.makedirs(prototype_files_dir, exist_ok=True)

    local_analysis_dataset_dir = os.path.join(
        local_analysis_dir,
        dataset_type,
    )
    os.makedirs(local_analysis_dataset_dir, exist_ok=True)

    log.info("Starting Testing")
    trainer.test(model=model, dataloaders=dataloader, ckpt_path=None)
    test_metrics = trainer.callback_metrics

    metric_dict = {**test_metrics}
    metric_dict = [
        {"name": k, "value": v.item() if hasattr(v, "item") else v}
        for k, v in metric_dict.items()
    ]

    file_path = os.path.join(local_analysis_dataset_dir, "finalmetrics.json")
    with open(file_path, "w") as json_file:
        json.dump(metric_dict, json_file)

    # Step 4: Check prototype identities within the network
    log.info("Checking prototype identities...")
    (
        prototype_labels_original,
        prototype_categories_original,
        prototype_max_connection_original,
    ) = check_prototype_identities(
        prototype_files_dir=prototype_files_dir,
        prototype_network=model,
        logger=log,
    )

    # Create a numpy array containing all indices of prototype_labels_original
    prototype_indices_original = np.arange(len(prototype_labels_original))

    # Extract the unique categories from the label to category mapping
    valid_categories = set(label_to_category_mapping.values())

    # Create a boolean mask where True indicates the prototype category is in the valid categories set
    prototype_mask_np = np.isin(prototype_categories_original, list(valid_categories))

    # Create a boolean mask as a list and convert it to a torch tensor
    prototype_mask_torch = torch.tensor(
        [category in valid_categories for category in prototype_categories_original],
        dtype=torch.bool,
    )

    # Use the mask to filter the prototype categories

    prototype_labels_original = prototype_labels_original[prototype_mask_np]
    prototype_categories = prototype_categories_original[prototype_mask_np]
    prototype_max_connection_masked = prototype_max_connection_original[
        prototype_mask_np
    ]
    prototype_indices_masked = prototype_indices_original[prototype_mask_np]

    # Step 1: Create a mapping from categories to labels from dataset B
    category_to_label_mapping = {v: k for k, v in label_to_category_mapping.items()}

    # Step 3: Build the mapping from true labels in dataset A to labels in dataset B using categories
    prototype_labels_to_dataset_labels_mapping = {
        prototype_labels_original[i]: category_to_label_mapping[prototype_categories[i]]
        for i in range(len(prototype_labels_original))
    }

    # Invert the mapping
    dataset_labels_to_prototype_labels_mapping = {
        value: key for key, value in prototype_labels_to_dataset_labels_mapping.items()
    }

    # Step 4: Convert true_labels_A and strongest_labels_A from dataset A to dataset B using the mapping
    prototype_labels = np.array(
        [
            prototype_labels_to_dataset_labels_mapping[label]
            for label in prototype_labels_original
        ]
    )
    prototype_max_connection = np.array(
        [
            prototype_labels_to_dataset_labels_mapping[label]
            for label in prototype_max_connection_masked
        ]
    )

    sample_rate = cfg.module.network.sampling_rate
    n_fft = cfg.datamodule.transforms.preprocessing.spectrogram_conversion.n_fft
    hop_length = (
        cfg.datamodule.transforms.preprocessing.spectrogram_conversion.hop_length
    )
    n_mels = cfg.datamodule.transforms.preprocessing.melscale_conversion.n_mels

    num_top_classes = cfg.local_analysis.number_top_classes
    max_samples = cfg.local_analysis.max_samples
    classification_threshold = cfg.local_analysis.classification_threshold

    save_prototype_spectrogram_files = cfg.save_prototype_spectrogram_files
    save_prototype_waveform_files = cfg.save_prototype_waveform_files

    # Step 7: Analyze each sample in the dataloader
    for sample_index, batch in enumerate(dataloader):
        # Check if max_samples is set and if the current index has reached the limit.
        if max_samples is not None and sample_index >= max_samples:
            log.info(
                f"Reached the limit of {max_samples} samples. Stop local analysis."
            )
            break

        log.info(f"Analyzing sample {sample_index} in {dataset_type} dataset...")

        # Determine directory path for saving analysis files for this sample
        local_analysis_sample_dir = os.path.join(
            local_analysis_dir,
            dataset_type,
            str(sample_index),
        )
        os.makedirs(local_analysis_sample_dir, exist_ok=True)

        spectrogram = batch["input_values"]
        spectrogram_height = spectrogram.shape[2]
        spectrogram_width = spectrogram.shape[3]

        # Find the indices where the elements are not zero
        labels = tuple((batch["labels"] != 0).nonzero(as_tuple=True)[1].tolist())
        categories = tuple(label_to_category_mapping[idx] for idx in labels)

        # Analyze the most activated prototypes for the sample
        (
            spectrogram_unnormalized_torch,
            spectrogram_unnormalized_np_without_channel_dim,
            logits,
            probabilities,
            classification_information,
            prototype_activations,
            prototype_activation_patterns,
        ) = analyze_most_activated_prototypes(
            prototype_network=model,
            label_to_category_mapping=label_to_category_mapping,
            prototype_labels=prototype_labels,
            prototype_labels_original=prototype_labels_original,
            prototype_categories=prototype_categories,
            prototype_max_connection=prototype_max_connection,
            prototype_indices_original=prototype_indices_masked,
            train_mean=datamodule.transforms.preprocessing.mean,
            train_std=datamodule.transforms.preprocessing.std,
            spectrogram_width=spectrogram_width,
            spectrogram_height=spectrogram_height,
            spectrogram=spectrogram,
            labels=labels,
            categories=categories,
            config=cfg,
            logger=log,
            local_analysis_dir=local_analysis_sample_dir,
            prototype_files_dir=prototype_files_dir,
            sample_rate=sample_rate,
            hop_length=hop_length,
            n_fft=n_fft,
            n_mels=n_mels,
            classification_threshold=classification_threshold,
            save_prototype_spectrogram_files=save_prototype_spectrogram_files,
            save_prototype_waveform_files=save_prototype_waveform_files,
            prototype_mask=prototype_mask_torch,
            dataset_labels_to_prototype_labels_mapping=dataset_labels_to_prototype_labels_mapping,
        )

        # Analyze the prototypes of the top K classes for the sample
        analyze_top_classes(
            prototype_network=model,
            prototype_labels=prototype_labels,
            prototype_labels_original=prototype_labels_original,
            prototype_categories=prototype_categories,
            prototype_activations=prototype_activations,
            prototype_activation_patterns=prototype_activation_patterns,
            prototype_max_connection=prototype_max_connection,
            prototype_indices_original=prototype_indices_masked,
            spectrogram_height=spectrogram_height,
            spectrogram_width=spectrogram_width,
            label_to_category_mapping=label_to_category_mapping,
            logits=logits,
            probabilities=probabilities,
            classification_information=classification_information,
            spectrogram_unnormalized_torch=spectrogram_unnormalized_torch,
            spectrogram_unnormalized_np_without_channel_dim=spectrogram_unnormalized_np_without_channel_dim,
            local_analysis_dir=local_analysis_sample_dir,
            logger=log,
            prototype_files_dir=prototype_files_dir,
            sample_rate=sample_rate,
            hop_length=hop_length,
            num_top_classes=num_top_classes,
            n_fft=n_fft,
            n_mels=n_mels,
            save_prototype_spectrogram_files=save_prototype_spectrogram_files,
            save_prototype_waveform_files=save_prototype_waveform_files,
        )

    log.info("Local analysis completed successfully.")


if __name__ == "__main__":
    local_analysis()
