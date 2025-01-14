from typing import Dict, List


def get_label_to_category_mapping_from_hf_dataset(
    dataset: List[Dict[str, str]]
) -> Dict[int, str]:
    """
    Get the mapping between labels in integer format and corresponding labels in string format.

    This function takes a Hugging Face dataset as input, where each sample is a dictionary containing 'target'
    (integer label) and 'category' (string category label). It extracts unique integer
    labels and their corresponding string category labels from the dataset and returns
    them as a dictionary.

    Args:
        dataset (List[Dict[str, str]]): The Hugging Face dataset containing samples with 'target' and 'category'.

    Returns:
        Dict[int, str]: A dictionary mapping integer labels to their corresponding string
        category labels.
    """
    label_to_category_mapping = {}

    # Iterate through each sample in the dataset
    for sample in dataset:
        # Convert the label to an integer
        label = int(sample["target"])
        category = sample["category"]

        # Check if the label is unique, and add it to the mapping
        if label not in label_to_category_mapping:
            label_to_category_mapping[label] = category

    return label_to_category_mapping
