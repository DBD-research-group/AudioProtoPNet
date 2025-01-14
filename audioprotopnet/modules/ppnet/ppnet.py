import math
from typing import Dict, List, Optional, Tuple
import warnings

import datasets
import torch
from torch import nn


class NonNegativeLinear(nn.Module):
    """
    A PyTorch module for a linear layer with non-negative weights.

    This module applies a linear transformation to the incoming data: `y = xA^T + b`.
    The weights of the transformation are constrained to be non-negative, making this
    module particularly useful in models where negative weights may not be appropriate.

    Attributes:
        in_features (int): The number of features in the input tensor.
        out_features (int): The number of features in the output tensor.
        weight (torch.Tensor): The weight parameter of the module, constrained to be non-negative.
        bias (torch.Tensor, optional): The bias parameter of the module.

    Args:
        in_features (int): The number of features in the input tensor.
        out_features (int): The number of features in the output tensor.
        bias (bool, optional): If True, the layer will include a learnable bias. Default: True.
        device (optional): The device (CPU/GPU) on which to perform computations.
        dtype (optional): The data type for the parameters (e.g., float32).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the NonNegativeLinear module.

        Args:
            input (torch.Tensor): The input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, out_features).
        """
        return nn.functional.linear(input, torch.relu(self.weight), self.bias)


class LinearLayerWithoutNegativeConnections(nn.Module):
    r"""
    Custom Linear Layer where each output class is connected to a specific subset of input features.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
        device: the device of the module parameters. Default: ``None``
        dtype: the data type of the module parameters. Default: ``None``

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out_features}, \text{features_per_output_class})`.
        bias: the learnable bias of the module of shape :math:`(\text{out_features})`.
              If :attr:`bias` is ``True``, the values are initialized from
              :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
              :math:`k = \frac{1}{\text{features_per_output_class}}`
    """

    __constants__ = ["in_features", "out_features", "bias"]
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        non_negative: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.non_negative = non_negative

        # Calculate the number of features per output class
        self.features_per_output_class = in_features // out_features

        # Ensure input size is divisible by the output size
        assert (
            in_features % out_features == 0
        ), "in_features must be divisible by out_features"

        # Define weights and biases
        self.weight = nn.Parameter(
            torch.empty(
                (out_features, self.features_per_output_class), **factory_kwargs
            )
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        # Initialize weights and biases
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initialize the weights and biases.
        Weights are initialized using Kaiming uniform initialization.
        Biases are initialized using a uniform distribution.
        """
        # Kaiming uniform initialization for the weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            # Calculate fan-in and fan-out values
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)

            # Uniform initialization for the biases
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the custom linear layer.

        Args:
            input (Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            Tensor: Output tensor of shape (batch_size, out_features).
        """
        batch_size = input.size(0)
        # Reshape input to (batch_size, out_features, features_per_output_class)
        reshaped_input = input.view(
            batch_size, self.out_features, self.features_per_output_class
        )

        # Apply ReLU to weights if non_negative_last_layer is True
        weight = torch.relu(self.weight) if self.non_negative else self.weight

        # Perform batch matrix multiplication and add bias
        output = torch.einsum("bof,of->bo", reshaped_input, weight)

        if self.bias is not None:
            output += self.bias

        return output

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class PPNet(nn.Module):
    def __init__(
        self,
        backbone_model: nn.Module,
        prototype_shape: Dict[str, int],
        num_classes: int,
        topk_k: int = 1,
        margin: Optional[float] = None,
        init_weights: bool = True,
        add_on_layers_type: str = "bottleneck",
        incorrect_class_connection: Optional[float] = -0.5,
        correct_class_connection: float = 1.0,
        bias_last_layer: Optional[float] = None,
        non_negative_last_layer: bool = True,
        embedded_spectrogram_height: Optional[int] = None,
        pretrain_info: Optional[Dict] = None,
    ) -> None:
        """
        PPNet is a class that implements the Prototypical Part Network (ProtoPNet) for prototype-based classification.

        Args:
            backbone_model (nn.Module): A PyTorch model to use as a base architecture for the PPNet.
            prototype_shape (Tuple[int, int, int, int]): A tuple representing the shape of the prototypes in the PPNet,
                as (number of prototypes, number of channels, prototype height, prototype width).
            num_classes (int): The number of classes in the classification task.
            topk_k (int): The number of top prototype activations to consider during training (default=1).
            margin (Optional[float]): The margin to use for subtractive margin cross entropy. (default=None).
            init_weights (bool): A boolean indicating whether to initialize the weights of the PPNet (default=True).
            add_on_layers_type (str): A string indicating the type of additional layers to add to the base architecture,
                can be 'bottleneck', 'identity', or 'upsample' (default='bottleneck').
            incorrect_class_connection (float): A float value incorrect class connections are initialized to
            (default=-1).

         Raises:
            ValueError: Raises an error if the number of prototypes is not evenly divisible by the number
             of classes, i.e. there are not the same number of prototypes for each class.
        """

        super().__init__()
        self.prototype_shape = (
            prototype_shape["num_prototypes"],
            prototype_shape["channels"],
            prototype_shape["height"],
            prototype_shape["width"],
        )
        self.num_prototypes = prototype_shape["num_prototypes"]
        self.num_prototypes_after_pruning = None
        self.margin = margin
        self.relu_on_cos = True
        self.incorrect_class_connection = incorrect_class_connection
        self.correct_class_connection = correct_class_connection
        self.input_vector_length = 64
        self.n_eps_channels = 2
        self.epsilon_val = 1e-4
        self.topk_k = topk_k
        self.bias_last_layer = bias_last_layer
        self.non_negative_last_layer = non_negative_last_layer
        self.embedded_spectrogram_height = embedded_spectrogram_height

        if self.bias_last_layer:
            self.use_bias_last_layer = True
        else:
            self.use_bias_last_layer = False

        self.prototype_class_identity = None
        self.num_prototypes_per_class = None

        if pretrain_info:
            self.pretrain_info = pretrain_info
            self.hf_path = pretrain_info.hf_path
            self.hf_name = (
                pretrain_info.hf_name
                if not pretrain_info.hf_pretrain_name
                else pretrain_info.hf_pretrain_name
            )
            self.num_classes = len(
                datasets.load_dataset_builder(self.hf_path, self.hf_name)
                .info.features["ebird_code"]
                .names
            )
        else:
            self.pretrain_info = None
            self.hf_path = None
            self.hf_name = None
            self.num_classes = num_classes

        # Checking the number of prototypes
        if self.num_prototypes % self.num_classes != 0:
            warnings.warn(
                "Number of prototypes is not evenly divisible by the number of classes.",
                UserWarning,
            )
        else:
            # Calculate the number of prototypes per class
            self.num_prototypes_per_class = self.num_prototypes // self.num_classes

            # Create a 1D tensor where each element represents the class index
            self.prototype_class_identity = (
                torch.arange(self.num_prototypes) // self.num_prototypes_per_class
            )

        # for j in range(self.num_prototypes):
        #    self.prototype_class_identity[j, j // self.num_prototypes_per_class] = 1

        # this has to be named backbone_model to allow the precise loading
        self.backbone_model = backbone_model

        self._setup_add_on_layers(add_on_layers_type=add_on_layers_type)

        self.prototype_vectors = nn.Parameter(
            torch.rand(self.prototype_shape), requires_grad=True
        )

        if self.embedded_spectrogram_height:
            # Initialize the frequency weights with a large positive value of 3.0 so that sigmoid(frequency_weights) is close to 1.
            self.frequency_weights = nn.Parameter(
                torch.full(
                    (
                        self.num_prototypes,
                        self.embedded_spectrogram_height,
                    ),
                    3.0,
                )
            )
        else:
            self.frequency_weights = None

        if self.incorrect_class_connection:
            if self.non_negative_last_layer:
                self.last_layer = NonNegativeLinear(
                    self.num_prototypes, self.num_classes, bias=self.use_bias_last_layer
                )
            else:
                self.last_layer = nn.Linear(
                    self.num_prototypes, self.num_classes, bias=self.use_bias_last_layer
                )
        else:
            self.last_layer = LinearLayerWithoutNegativeConnections(
                in_features=self.num_prototypes,
                out_features=self.num_classes,
                non_negative=self.non_negative_last_layer,
            )

        if init_weights:
            self._initialize_weights()

    def _setup_add_on_layers(self, add_on_layers_type: str):
        """
        Configures additional layers based on the backbone model architecture and the specified add_on_layers_type.

        Args:
            add_on_layers_type (str): Type of additional layers to add. Can be 'identity' or 'upsample'.
        """

        if add_on_layers_type == "identity":
            self.add_on_layers = nn.Sequential(nn.Identity())
        elif add_on_layers_type == "upsample":
            self.add_on_layers = nn.Upsample(scale_factor=2, mode="bilinear")
        else:
            raise NotImplementedError(
                f"The add-on layer type {add_on_layers_type} isn't implemented yet."
            )

    def conv_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Takes an input tensor and passes it through the backbone model to extract features.
        Then, it passes them through the additional layers to produce the output tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after passing through the backbone model and additional layers.
        """
        # Extract features using the backbone model
        features = self.backbone_model(x)

        # The features must be a 4D tensor of shape (batch size, channels, height, width)
        if features.dim() == 3:
            features.unsqueeze_(0)

        # Pass the features through additional layers
        output = self.add_on_layers(features)

        return output

    def cos_activation(
        self,
        x: torch.Tensor,
        prototypes_of_wrong_class: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the cosine activation between input tensor x and prototype vectors.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor with shape (batch_size, num_channels, height, width).
        prototypes_of_wrong_class : Optional[torch.Tensor]
            Tensor containing the prototypes of the wrong class with shape (batch_size, num_prototypes).

        Returns:
        --------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing:
            - activations: The cosine activations with potential margin adjustments.
            - marginless_activations: The cosine activations without margin adjustments.
        """
        input_vector_length = self.input_vector_length
        normalizing_factor = (
            self.prototype_shape[-2] * self.prototype_shape[-1]
        ) ** 0.5

        # Pre-allocate epsilon channels on the correct device for input tensor x
        epsilon_channel_x = torch.full(
            (x.shape[0], self.n_eps_channels, x.shape[2], x.shape[3]),
            self.epsilon_val,
            device=x.device,
            requires_grad=False,
        )
        x = torch.cat((x, epsilon_channel_x), dim=-3)

        # Normalize x
        x_length = torch.sqrt(torch.sum(x**2, dim=-3, keepdim=True) + self.epsilon_val)
        x_normalized = (input_vector_length * x / x_length) / normalizing_factor

        # Pre-allocate epsilon channels for prototypes on the correct device
        epsilon_channel_p = torch.full(
            (
                self.prototype_shape[0],
                self.n_eps_channels,
                self.prototype_shape[2],
                self.prototype_shape[3],
            ),
            self.epsilon_val,
            device=self.prototype_vectors.device,
            requires_grad=False,
        )
        appended_protos = torch.cat((self.prototype_vectors, epsilon_channel_p), dim=-3)

        # Normalize prototypes
        prototype_vector_length = torch.sqrt(
            torch.sum(appended_protos**2, dim=-3, keepdim=True) + self.epsilon_val
        )
        normalized_prototypes = appended_protos / (
            prototype_vector_length + self.epsilon_val
        )
        normalized_prototypes /= normalizing_factor

        # Compute activations using convolution
        activations_dot = nn.functional.conv2d(x_normalized, normalized_prototypes)
        marginless_activations = activations_dot / (input_vector_length * 1.01)

        if self.frequency_weights is not None:
            # Apply sigmoid to frequency weights. s.t. weights are between 0 and 1.
            freq_weights = torch.sigmoid(self.frequency_weights)

            # Multiply each prototype's frequency response by the corresponding weights
            marginless_activations = marginless_activations * freq_weights[:, :, None]

        if (
            self.margin is None
            or not self.training
            or prototypes_of_wrong_class is None
        ):
            activations = marginless_activations
        else:
            # Apply margin adjustment for wrong class prototypes
            wrong_class_margin = (prototypes_of_wrong_class * self.margin).view(
                x.size(0), self.prototype_vectors.size(0), 1, 1
            )
            wrong_class_margin = wrong_class_margin.expand(
                -1, -1, activations_dot.size(-2), activations_dot.size(-1)
            )
            penalized_angles = (
                torch.acos(activations_dot / (input_vector_length * 1.01))
                - wrong_class_margin
            )
            activations = torch.cos(torch.relu(penalized_angles))

        if self.relu_on_cos:
            # Apply ReLU activation on the cosine values
            activations = torch.relu(activations)
            marginless_activations = torch.relu(marginless_activations)

        return activations, marginless_activations

    def prototype_activations(
        self,
        x: torch.Tensor,
        prototypes_of_wrong_class: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Compute the prototype activations for a given input tensor.

        Args:
        - x (torch.Tensor): The raw input tensor with shape (batch_size, num_channels, height, width).
        - prototypes_of_wrong_class (Optional[torch.Tensor]): The prototypes of the wrong classes that are needed
            when using subtractive margins. Defaults to None.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]:
            - activations: A tensor containing the prototype activations.
            - a list containing:
                - marginless_activations: A tensor containing the activations before applying subtractive margin.
                - conv_features: A tensor containing the convolutional features.
        """
        # Compute convolutional features
        conv_features = self.conv_features(x)

        # Compute cosine activations
        activations, marginless_activations = self.cos_activation(
            conv_features,
            prototypes_of_wrong_class=prototypes_of_wrong_class,
        )

        return activations, [marginless_activations, conv_features]

    def forward(
        self,
        x: torch.Tensor,
        prototypes_of_wrong_class: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass of the PPNet model.

        Args:
        - x (torch.Tensor): Input tensor with shape (batch_size, num_channels, height, width).
        - prototypes_of_wrong_class (Optional[torch.Tensor]): The prototypes of the wrong classes that are needed
            when using subtractive margins. Defaults to None.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]:
            - logits: A tensor containing the logits for each class in the model.
            - a list containing:
                - mean_activations: A tensor containing the mean of the top-k prototype activations.
                (in evaluation mode k is always 1)
                - marginless_logits: A tensor containing the logits for each class in the model, calculated using the
                marginless activations.
                - conv_features: A tensor containing the convolutional features.
                - marginless_max_activations: A tensor containing the max-pooled marginless activations.

        """
        activations, additional_returns = self.prototype_activations(
            x, prototypes_of_wrong_class=prototypes_of_wrong_class
        )
        marginless_activations = additional_returns[0]
        conv_features = additional_returns[1]

        # Set topk_k based on training mode: use predefined value if training, else 1 for evaluation
        if self.training:
            topk_k = self.topk_k
        else:
            topk_k = 1

        # Reshape activations to combine spatial dimensions: (batch_size, num_prototypes, height*width)
        activations = activations.view(activations.shape[0], activations.shape[1], -1)

        # Perform top-k pooling along the combined spatial dimension
        # For topk_k=1, this is equivalent to global max pooling
        topk_activations, _ = torch.topk(activations, topk_k, dim=-1)

        # Calculate the mean of the top-k activations for each channel: (batch_size, num_channels)
        # If topk_k=1, this mean operation does nothing since there's only one value.
        mean_activations = torch.mean(topk_activations, dim=-1)

        marginless_max_activations = nn.functional.max_pool2d(
            marginless_activations,
            kernel_size=(
                marginless_activations.size()[2],
                marginless_activations.size()[3],
            ),
        )
        marginless_max_activations = marginless_max_activations.view(
            -1, self.num_prototypes
        )

        logits = self.last_layer(mean_activations)
        marginless_logits = self.last_layer(marginless_max_activations)
        return logits, [
            mean_activations,
            marginless_logits,
            conv_features,
            marginless_max_activations,
            marginless_activations,
        ]

    def push_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This method is needed for the pushing operation.

        Args:
        - x (torch.Tensor): Input tensor of shape (batch_size, num_channels, height, width).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the convolutional features and marginless activations.
        """

        conv_output = self.conv_features(x)
        _, marginless_activations = self.cos_activation(conv_output)
        return conv_output, marginless_activations

    def get_prototype_orthogonalities(
        self, use_part_prototypes: bool = False
    ) -> torch.Tensor:
        """
        Computes the orthogonality loss, encouraging each piece of a prototype to be orthogonal to the others.

        This method is inspired by the paper:
        https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Interpretable_Image_Recognition_by_Constructing_Transparent_Embedding_Space_ICCV_2021_paper.pdf

        Args:
            use_part_prototypes (bool): If True, treats each spatial part of the prototypes as a separate prototype.

        Returns:
            torch.Tensor: A tensor representing the orthogonalities.
        """

        if use_part_prototypes:
            # Normalize prototypes to unit length
            prototype_vector_length = torch.sqrt(
                torch.sum(torch.square(self.prototype_vectors), dim=1, keepdim=True)
                + self.epsilon_val
            )
            normalized_prototypes = self.prototype_vectors / (
                prototype_vector_length + self.epsilon_val
            )

            # Calculate total part prototypes per class
            num_part_prototypes_per_class = (
                self.num_prototypes_per_class
                * self.prototype_shape[2]
                * self.prototype_shape[3]
            )

            # Reshape to match class structure
            normalized_prototypes = normalized_prototypes.view(
                self.num_classes,
                self.num_prototypes_per_class,
                self.prototype_shape[1],
                self.prototype_shape[2] * self.prototype_shape[3],
            )

            # Transpose and reshape to treat each spatial part as a separate prototype
            normalized_prototypes = normalized_prototypes.permute(0, 1, 3, 2).reshape(
                self.num_classes, num_part_prototypes_per_class, self.prototype_shape[1]
            )

        else:
            # Normalize prototypes to unit length
            prototype_vectors_reshaped = self.prototype_vectors.view(
                self.num_prototypes, -1
            )
            prototype_vector_length = torch.sqrt(
                torch.sum(torch.square(prototype_vectors_reshaped), dim=1, keepdim=True)
                + self.epsilon_val
            )
            normalized_prototypes = prototype_vectors_reshaped / (
                prototype_vector_length + self.epsilon_val
            )

            # Reshape to match class structure
            normalized_prototypes = normalized_prototypes.view(
                self.num_classes,
                self.num_prototypes_per_class,
                self.prototype_shape[1]
                * self.prototype_shape[2]
                * self.prototype_shape[3],
            )

        # Compute orthogonality matrix for each class
        orthogonalities = torch.matmul(
            normalized_prototypes, normalized_prototypes.transpose(1, 2)
        )

        # Identity matrix to enforce orthogonality
        identity_matrix = (
            torch.eye(normalized_prototypes.shape[1], device=orthogonalities.device)
            .unsqueeze(0)
            .repeat(self.num_classes, 1, 1)
        )

        # Subtract identity to focus on orthogonality
        orthogonalities = orthogonalities - identity_matrix

        return orthogonalities

    def identify_prototypes_to_prune(self) -> List[int]:
        """
        Identifies the indices of prototypes that should be pruned.

        This function iterates through the prototypes and checks if the specific weight
        connecting the prototype to its class is zero. It is specifically designed to handle
        the LinearLayerWithoutNegativeConnections where each class has a subset of features
        it connects to.

        Returns:
            List[int]: A list of prototype indices that should be pruned.
        """
        prototypes_to_prune = []

        # Calculate the number of prototypes assigned to each class
        prototypes_per_class = self.num_prototypes // self.num_classes

        if isinstance(self.last_layer, LinearLayerWithoutNegativeConnections):
            # Custom layer mapping prototypes to a subset of input features for each output class
            for prototype_index in range(self.num_prototypes):
                class_index = self.prototype_class_identity[prototype_index]
                # Calculate the specific index within the 'features_per_output_class' for this prototype
                index_within_class = prototype_index % prototypes_per_class
                # Check if the specific weight connecting the prototype to its class is zero
                if self.last_layer.weight[class_index, index_within_class] == 0.0:
                    prototypes_to_prune.append(prototype_index)
        else:
            # Standard linear layer: each prototype directly maps to a feature index
            weights_to_check = self.last_layer.weight
            for prototype_index in range(self.num_prototypes):
                class_index = self.prototype_class_identity[prototype_index]
                if weights_to_check[class_index, prototype_index] == 0.0:
                    prototypes_to_prune.append(prototype_index)

        return prototypes_to_prune

    def prune_prototypes_by_threshold(self, threshold: float = 1e-3) -> None:
        """
        Prune the weights in the classification layer by setting weights below a specified threshold to zero.

        This method modifies the weights of the last layer of the model in-place. Weights falling below the
        threshold are set to zero, diminishing their influence in the model's decisions. It also identifies
        and prunes prototypes based on these updated weights, thereby refining the model's structure.

        Args:
            threshold (float): The threshold value below which weights will be set to zero. Defaults to 1e-3.
        """
        # Access the weights of the last layer
        weights = self.last_layer.weight.data

        # Set weights below the threshold to zero
        # This step reduces the influence of low-value weights in the model's decision-making process
        weights[weights < threshold] = 0.0

        # Update the weights in the last layer to reflect the pruning
        self.last_layer.weight.data.copy_(weights)

        # Identify prototypes that need to be pruned based on the updated weights
        prototypes_to_prune = self.identify_prototypes_to_prune()

        # Execute the pruning of identified prototypes
        self.prune_prototypes_by_index(prototypes_to_prune)

    def prune_prototypes_by_index(self, prototypes_to_prune: List[int]) -> None:
        """
        Prunes specified prototypes from the PPNet.

        Args:
            prototypes_to_prune (List[int]): A list of indices indicating the prototypes to be removed.
                                             Each index should be in the range [0, current number of prototypes - 1].

        Returns:
            None
        """

        # Validate the provided indices to ensure they are within the valid range
        if any(
            index < 0 or index >= self.num_prototypes for index in prototypes_to_prune
        ):
            raise ValueError("Provided prototype indices are out of valid range!")

        # Calculate the new number of prototypes after pruning
        self.num_prototypes_after_pruning = self.num_prototypes - len(
            prototypes_to_prune
        )

        # Remove the prototype vectors that are no longer needed
        with torch.no_grad():
            # If frequency_weights are being used, set the weights of pruned prototypes to -7
            if self.frequency_weights is not None:
                self.frequency_weights.data[prototypes_to_prune, :] = -7.0

            # Adjust the weights in the last layer depending on its type
            if isinstance(self.last_layer, LinearLayerWithoutNegativeConnections):
                # For LinearLayerWithoutNegativeConnections, set the connection weights to zero
                # only for the pruned prototypes related to their specific classes
                for class_idx in range(self.last_layer.out_features):
                    # Identify prototypes belonging to the current class
                    indices_for_class = [
                        idx % self.last_layer.features_per_output_class
                        for idx in prototypes_to_prune
                        if self.prototype_class_identity[idx] == class_idx
                    ]
                    self.last_layer.weight.data[class_idx, indices_for_class] = 0.0
            else:
                # For other layer types, set the weights of pruned prototypes to zero
                self.last_layer.weight.data[:, prototypes_to_prune] = 0.0

    def __repr__(self) -> str:
        """
        Returns a string that describes the PPNet object.

        Returns:
            str: A string representation of the PPNet object.
        """
        rep = f"""PPNet(
        backbone_model: {self.backbone_model},
        prototype_shape: {self.prototype_shape},
        num_classes: {self.num_classes},
        epsilon: {self.epsilon_val})"""

        return rep

    def set_last_layer_incorrect_connection(
        self, incorrect_strength: Optional[float] = None
    ) -> None:
        """
        Modifies the last layer weights to have incorrect connections with a specified strength.
        If incorrect_strength is None, initializes the weights for LinearLayerWithoutNegativeConnections
        with correct_class_connection value.

        Args:
        - incorrect_strength (Optional[float]): The strength of the incorrect connections.
                                                If None, initialize without incorrect connections.

        Returns:
            None
        """
        if incorrect_strength is None:
            # Handle LinearLayerWithoutNegativeConnections initialization
            if isinstance(self.last_layer, LinearLayerWithoutNegativeConnections):
                # Initialize all weights to the correct_class_connection value
                self.last_layer.weight.data.fill_(self.correct_class_connection)
            else:
                raise ValueError(
                    "last_layer is not an instance of LinearLayerWithoutNegativeConnections"
                )

        else:
            # Create a one-hot matrix for correct connections
            positive_one_weights_locations = torch.zeros(
                self.num_classes, self.num_prototypes
            )
            positive_one_weights_locations[
                self.prototype_class_identity,
                torch.arange(self.num_prototypes),
            ] = 1

            # Create a matrix for incorrect connections
            negative_one_weights_locations = 1 - positive_one_weights_locations

            # This variable represents the strength of the connection for correct class
            correct_class_connection = self.correct_class_connection

            # This variable represents the strength of the connection for incorrect class
            incorrect_class_connection = incorrect_strength

            # Modify weights to have correct and incorrect connections
            self.last_layer.weight.data.copy_(
                correct_class_connection * positive_one_weights_locations
                + incorrect_class_connection * negative_one_weights_locations
            )

        if self.last_layer.bias is not None:
            # Initialize all biases to bias_last_layer value
            self.last_layer.bias.data.fill_(self.bias_last_layer)

    def _initialize_weights(self) -> None:
        """
        Initializes the weights of the add-on layers of the network and the last layer with incorrect connections.

        Returns:
            None
        """

        for m in self.add_on_layers.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Initialize the last layer with incorrect connections using specified incorrect class connection strength
        self.set_last_layer_incorrect_connection(
            incorrect_strength=self.incorrect_class_connection
        )
