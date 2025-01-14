from typing import Tuple

import torch
from torch.utils.data import DataLoader


def calculate_mean_std_from_dataloader(dataloader: DataLoader) -> Tuple[float, float]:
    """
    Calculates the mean and sample standard deviation from a PyTorch DataLoader.

    This function iteratively processes each batch from the DataLoader, accumulating the sum of all input values
    and the sum of their squares to compute the dataset's mean and sample standard deviation. It uses Bessel's
    correction (dividing by N-1) for the standard deviation to provide an unbiased estimator when the data is
    a sample of a larger population.

    Args:
        dataloader (DataLoader): The DataLoader containing the dataset for which statistics are to be computed.

    Returns:
        Tuple[float, float]: A tuple containing the mean and sample standard deviation of the dataset.

    Raises:
        ValueError: If the DataLoader is empty and no elements can be used for computation.
    """

    sum_, sum_of_squares, num_elements = 0.0, 0.0, 0

    for batch in dataloader:
        # Assume 'input_values' is a tensor containing the relevant data
        input_values = batch["input_values"]
        sum_ += (
            input_values.sum().item()
        )  # Accumulate the sum of all elements in this batch
        sum_of_squares += (
            (input_values**2).sum().item()
        )  # Sum of squares for the current batch
        num_elements += (
            input_values.nelement()
        )  # Count the number of elements in this batch

    if num_elements == 0:
        raise ValueError(
            "DataLoader is empty. Cannot compute mean and standard deviation."
        )

    mean = sum_ / num_elements  # Calculate the mean of the dataset
    # Calculate the sample standard deviation using Bessel's correction (n-1 in the denominator)
    std = ((sum_of_squares - sum_**2 / num_elements) / (num_elements - 1)) ** 0.5

    return mean, std


def standardize_tensor(x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """
    Standardizes a tensor by subtracting the mean and dividing by the standard deviation.

    This function applies element-wise normalization on the tensor to ensure that the resulting tensor
    has properties of a standard normal distribution with mean 0 and standard deviation 1, based on
    the provided `mean` and `std` parameters.

    Args:
        x (torch.Tensor): The input tensor to be standardized.
        mean (float): The mean used for standardizing.
        std (float): The standard deviation used for standardizing.

    Returns:
        torch.Tensor: The standardized tensor.

    Raises:
        ZeroDivisionError: If `std` is zero.
    """
    if std == 0:
        raise ZeroDivisionError(
            "Standard deviation must not be zero for standardization."
        )

    # Subtract the mean from each element and divide by the standard deviation for standardization
    x_standardized = (x - mean) / std

    return x_standardized


def undo_standardize_tensor(x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """
    Reverses the standardization applied by the `standardize_tensor` function.

    This function restores a standardized tensor to its original scale by multiplying by the
    standard deviation and adding the mean. This is useful for operations where the original scale of
    data is needed after standardization.

    Args:
        x (torch.Tensor): The standardized tensor to be restored.
        mean (float): The mean value used during the original standardization.
        std (float): The standard deviation used during the original standardization.

    Returns:
        torch.Tensor: The tensor restored to its original scale.

    Raises:
        ZeroDivisionError: If `std` is zero.
    """
    if std == 0:
        raise ZeroDivisionError(
            "Standard deviation must not be zero for reversing standardization."
        )

    # Multiply by the standard deviation and add the mean to restore the original scaling
    x_original = x * std + mean

    return x_original
