import math
from typing import Dict, List, Optional, Tuple

import hydra
import torch

from audioprotopnet.modules.base_module import BaseModule
from audioprotopnet.modules.ppnet.ppnet import LinearLayerWithoutNegativeConnections
from audioprotopnet.modules.ppnet.push import push_prototypes
from audioprotopnet.modules.ppnet.training_modes import (
    freeze_all_layers,
    joint,
    last_only,
    warm_only,
)


class LightningProtoPNet(BaseModule):
    def __init__(
        self,
        network,
        output_activation,
        loss,
        optimizer,
        metrics,
        num_epochs,
        batch_size,
        len_trainset,
        task,
        class_weights_loss,
        label_counts,
        num_gpus,
        training_phase: str,
        learning_rates: dict,
        lr_scheduler,
        prototype_files_dir: str,
        logging_params,
        label_to_category_mapping: Dict[int, str],
        train_mean: float,
        train_std: float,
        last_layer_fixed: bool,
        freeze_incorrect_class_connections: bool,
        subtractive_margin: bool,
        coefficients: Dict,
        weight_decay: float,
        prototype_layer_stride: int,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        n_mels: int,
        prediction_table=None,
    ) -> None:
        """
        A PyTorch Lightning wrapper for PPNet.

        Args:
            model (PPNet): An instance of the PPNet model.
            config (Dict[str, any]): Configuration settings.
            training_phase (str): The phase of training ("warm", "joint", "joint_with_last_layer", "push", "last_layer",
             "pruning", "push_final" or  "inference").
            weight_decay (float): Weight decay for optimizer.
            learning_rates (dict): Learning rate configurations.
            last_layer_fixed (bool): If true, the last layer's weights are fixed.
            coefficients (dict): Coefficients for various loss components.
            train_push_loader (torch.utils.data.DataLoader): DataLoader for pushing operation.
            prototype_layer_stride (int): Stride for the prototype layer.
            prototype_files_dir (str): Directory path for prototype files.
            sample_rate (int): Sample rate for audio processing.
            n_fft (int): FFT size for audio processing.
            hop_length (int): Hop length for audio processing.
            train_mean (float): Mean value for data normalization.
            train_std (float): Standard deviation value for data normalization.
            n_mels (int): Number of mel filters.
            subtractive_margin (bool): If true, use subtractive margin. Defaults to True.

        Raises:
            NotImplementedError: If the selected training phase is not supported.
        """
        super().__init__(
            network=network,
            output_activation=output_activation,
            loss=loss,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            metrics=metrics,
            logging_params=logging_params,
            num_epochs=num_epochs,
            len_trainset=len_trainset,
            task=task,
            class_weights_loss=class_weights_loss,
            label_counts=label_counts,
            batch_size=batch_size,
            num_gpus=num_gpus,
            prediction_table=prediction_table,
        )

        self.training_phase = training_phase
        self.learning_rates = learning_rates
        self.last_layer_fixed = last_layer_fixed
        self.freeze_incorrect_class_connections = freeze_incorrect_class_connections
        self.subtractive_margin = subtractive_margin
        self.coefficients = coefficients

        self.weight_decay = weight_decay

        self.prototype_layer_stride = prototype_layer_stride
        self.prototype_files_dir = prototype_files_dir
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.label_to_category_mapping = label_to_category_mapping
        self.train_mean = train_mean
        self.train_std = train_std
        self.n_mels = n_mels

        if self.training_phase == "warm":
            warm_only(model=self.model, last_layer_fixed=self.last_layer_fixed)
        elif self.training_phase == "joint":
            joint(model=self.model, last_layer_fixed=True)

        elif self.training_phase == "joint_with_last_layer":
            joint(
                model=self.model,
                last_layer_fixed=False,
                freeze_incorrect_class_connections=self.freeze_incorrect_class_connections,
            )
        elif self.training_phase == "last_layer":
            last_only(
                model=self.model,
                freeze_incorrect_class_connections=self.freeze_incorrect_class_connections,
            )
        elif self.training_phase in (
            "inference",
            "pruning",
            "push_warm",
            "push_joint",
            "push_joint_with_last_layer",
            "push_final",
        ):
            freeze_all_layers(model=self.model)
        else:
            raise NotImplementedError(
                f"The selected training phase {self.training_phase} is not supported."
            )

    def forward(self, batch: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Defines the forward pass, i.e., the computation performed on each call.

        Args:
            batch (torch.Tensor): A batch of input data.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: The output produced by the PPNet model.
        """
        return self.model(batch, prototypes_of_wrong_class=None)

    def configure_optimizers(self):
        optimizer = self.construct_optimizer()

        if self.lrs_params.get("scheduler"):
            num_training_steps = math.ceil(
                (self.num_epochs * self.len_trainset) / self.batch_size * self.num_gpus
            )

            scheduler_target = self.lrs_params.scheduler._target_
            is_linear_warmup = (
                scheduler_target == "transformers.get_linear_schedule_with_warmup"
            )
            is_cosine_warmup = (
                scheduler_target == "transformers.get_cosine_schedule_with_warmup"
            )

            if is_linear_warmup or is_cosine_warmup:
                num_warmup_steps = math.ceil(
                    num_training_steps * self.lrs_params.extras.warmup_ratio
                )

                scheduler_args = {
                    "optimizer": optimizer,
                    "num_warmup_steps": num_warmup_steps,
                    "num_training_steps": num_training_steps,
                    "_convert_": "partial",
                }
            else:
                scheduler_args = {"optimizer": optimizer, "_convert_": "partial"}

            # instantiate hydra
            scheduler = hydra.utils.instantiate(
                self.lrs_params.scheduler, **scheduler_args
            )
            lr_scheduler_dict = {"scheduler": scheduler}

            if self.lrs_params.get("extras"):
                for key, value in self.lrs_params.get("extras").items():
                    lr_scheduler_dict[key] = value

            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_dict}

        return {"optimizer": optimizer}

    def model_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Shared logic for training, validation and test steps.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): A tuple containing input data,
                corresponding labels, and categories.
            phase (str): Specifies the current phase, can be either "train", "validation", or "test".

        Returns:
            torch.Tensor: The computed loss.
        """

        inputs, labels = batch["input_values"], batch["labels"]

        if self.training_phase == "inference":
            logits, additional_returns = self.model(
                inputs, prototypes_of_wrong_class=None
            )

            if hasattr(self, "class_mask") and self.class_mask:
                logits = logits[:, self.class_mask]

            # Compute cross entropy loss
            cross_entropy_loss = self.loss(logits, labels)

            predictions = self.output_activation(logits)
            return cross_entropy_loss, predictions, labels
        else:
            if self.task == "multiclass":
                # Get the prototypes of the correct class for multiclass
                prototypes_of_correct_class = (
                    self.model.prototype_class_identity.unsqueeze(0).to(self.device)
                    == labels.unsqueeze(1)
                ).float()
            elif self.task == "multilabel":
                batch_size = labels.shape[0]

                # Expand prototype_class_identity to match the batch size
                expanded_prototype_class_identity = (
                    self.model.prototype_class_identity.expand(batch_size, -1)
                )

                # Use the class presence mask to index the prototypes of the correct class
                prototypes_of_correct_class = labels.gather(
                    1, expanded_prototype_class_identity.to(self.device)
                ).float()

            else:
                raise NotImplementedError(
                    f"Only the multiclass and multilabel tasks are implemented, not task {self.task}."
                )

            # Calculate the prototypes of the wrong class.
            prototypes_of_wrong_class = 1 - prototypes_of_correct_class

            if self.subtractive_margin:
                logits, additional_returns = self.model(
                    inputs,
                    prototypes_of_wrong_class=prototypes_of_wrong_class,
                )
            else:
                logits, additional_returns = self.model(
                    inputs, prototypes_of_wrong_class=None
                )

            if (
                hasattr(self, "class_mask")
                and self.class_mask
                and (
                    not self.model.pretrain_info.valid_test_only
                    or not self.trainer.training
                )
            ):
                logits = logits[:, self.class_mask]

            max_activations = additional_returns[0]

            # Compute cross entropy loss
            cross_entropy_loss = self.loss(logits, labels)

            if self.coefficients["cluster_cost"] is not None:
                # Compute cluster cost
                cluster_cost = self.calculate_cluster_cost(
                    max_activations=max_activations,
                    prototypes_of_correct_class=prototypes_of_correct_class,
                )
            else:
                cluster_cost = None

            if self.coefficients["separation_cost"] is not None:
                # Compute separation cost
                separation_cost = self.calculate_separation_cost(
                    max_activations=max_activations,
                    prototypes_of_wrong_class=prototypes_of_wrong_class,
                )
            else:
                separation_cost = None

            if self.coefficients["l1_loss_last_layer"] is not None:
                # Compute L1 loss for last layer
                l1_loss_last_layer = self.calculate_l1_loss_last_layer()
            else:
                l1_loss_last_layer = None

            if self.coefficients["zero_weights_loss_last_layer"] is not None:
                # Compute zero weights loss for last layer
                zero_weights_loss_last_layer = (
                    self.calculate_zero_weights_loss_last_layer()
                )
            else:
                zero_weights_loss_last_layer = None

            # Compute orthogonality loss if applicable
            if self.coefficients["orthogonality_loss"] is not None:
                # Compute keypoint-wise orthogonality loss
                orthogonality_loss = self.calculate_orthogonality_loss()
            else:
                # Set orthogonality loss to None
                orthogonality_loss = None

            # Compute L1 loss for frequency weights
            if (
                self.model.frequency_weights is not None
                and self.coefficients["l1_loss_frequency_weights"] is not None
            ):
                l1_loss_frequency_weights = self.calculate_l1_loss_frequency_weights()
            else:
                l1_loss_frequency_weights = None

            # Calculate the total loss
            loss = self.calculate_total_loss(
                cross_entropy_loss=cross_entropy_loss,
                cluster_cost=cluster_cost,
                separation_cost=separation_cost,
                l1_loss_last_layer=l1_loss_last_layer,
                zero_weights_loss_last_layer=zero_weights_loss_last_layer,
                l1_loss_frequency_weights=l1_loss_frequency_weights,
                orthogonality_loss=orthogonality_loss,
            )

            predictions = self.output_activation(logits)

            return loss, predictions, labels

    def construct_optimizer(self) -> torch.optim.Optimizer:
        """
        Constructs and returns an optimizer based on the training phase.

        Returns:
            torch.optim.Optimizer: The constructed optimizer.
        """
        if self.training_phase == "warm":
            optimizer_specifications = [
                {
                    "params": self.model.add_on_layers.parameters(),
                    "lr": self.learning_rates["warm"]["add_on_layers"],
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": self.model.prototype_vectors,
                    "lr": self.learning_rates["warm"]["prototype_vectors"],
                },
            ]

            if not self.last_layer_fixed:
                optimizer_specifications.append(
                    {
                        "params": self.model.last_layer.parameters(),
                        "lr": self.learning_rates["warm"]["warm_last_layer_lr"],
                        "weight_decay": self.weight_decay,
                    },
                )

            if self.model.frequency_weights is not None:
                optimizer_specifications.append(
                    {
                        "params": self.model.frequency_weights,
                        "lr": self.learning_rates["warm"]["frequency_weights"],
                    }
                )

        elif self.training_phase == "joint":
            optimizer_specifications = [
                {
                    "params": self.model.backbone_model.parameters(),
                    "lr": self.learning_rates["joint"]["backbone_model"],
                    "weight_decay": self.weight_decay,
                },  # bias are now also being regularized
                {
                    "params": self.model.add_on_layers.parameters(),
                    "lr": self.learning_rates["joint"]["add_on_layers"],
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": self.model.prototype_vectors,
                    "lr": self.learning_rates["joint"]["prototype_vectors"],
                },
            ]

            if self.model.frequency_weights is not None:
                optimizer_specifications.append(
                    {
                        "params": self.model.frequency_weights,
                        "lr": self.learning_rates["joint"]["frequency_weights"],
                    }
                )

        # TODO: Find a better name for this phase
        elif self.training_phase == "joint_with_last_layer":
            optimizer_specifications = [
                {
                    "params": self.model.backbone_model.parameters(),
                    "lr": self.learning_rates["joint"]["backbone_model"],
                    "weight_decay": self.weight_decay,
                },  # bias are now also being regularized
                {
                    "params": self.model.add_on_layers.parameters(),
                    "lr": self.learning_rates["joint"]["add_on_layers"],
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": self.model.prototype_vectors,
                    "lr": self.learning_rates["joint"]["prototype_vectors"],
                },
                {
                    "params": self.model.last_layer.parameters(),
                    "lr": self.learning_rates["joint"]["joint_last_layer_lr"],
                    "weight_decay": self.weight_decay,
                },
            ]

            if self.model.frequency_weights is not None:
                optimizer_specifications.append(
                    {
                        "params": self.model.frequency_weights,
                        "lr": self.learning_rates["joint"]["frequency_weights"],
                    }
                )

        elif self.training_phase == "last_layer":
            optimizer_specifications = [
                {
                    "params": self.model.last_layer.parameters(),
                    "lr": self.learning_rates["last_layer"],
                    "weight_decay": self.weight_decay,
                }
            ]

        elif self.training_phase == "inference":
            return None  # No optimizer is required for inference

        else:
            raise NotImplementedError("The selected training phase is not supported.")

        optimizer = hydra.utils.instantiate(
            self.opt_params, params=optimizer_specifications, _convert_="partial"
        )

        return optimizer

    def calculate_cluster_cost(
        self,
        max_activations: torch.Tensor,
        prototypes_of_correct_class: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculate the cluster cost based on the given activations and prototypes.

        Parameters:
        - max_activations (torch.Tensor): A tensor containing the maximum activations for each instance.
        - prototypes_of_correct_class (Optional[torch.Tensor]): A tensor indicating the prototypes of the correct class. Default is None.

        Returns:
        - Tensor: The calculated cluster cost.
        """
        device = max_activations.device  # Ensure all tensors are on the same device

        if self.task == "multiclass":
            # Multiclass classification case
            if prototypes_of_correct_class is not None:
                prototypes_of_correct_class = prototypes_of_correct_class.to(device)
                # If prototypes of the correct class are provided
                correct_class_prototype_activations, _ = torch.max(
                    max_activations * prototypes_of_correct_class, dim=1
                )
                cluster_cost = torch.mean(correct_class_prototype_activations)
            else:
                # If prototypes of the correct class are not provided
                maximum_max_activations, _ = torch.max(max_activations, dim=1)
                cluster_cost = torch.mean(maximum_max_activations)

        elif self.task == "multilabel":
            # Multilabel classification case
            if prototypes_of_correct_class is not None:
                prototypes_of_correct_class = prototypes_of_correct_class.to(device)
                # If prototypes of the correct class are provided
                correct_class_activations = (
                    max_activations * prototypes_of_correct_class
                )  # Shape: (num_instances, num_prototypes)

                # Create a mask for relevant classes for each instance
                relevant_classes_mask = (
                    prototypes_of_correct_class > 0
                )  # Shape: (num_instances, num_prototypes)

                # Map the prototype indices to their corresponding class indices
                class_indices = (
                    self.model.prototype_class_identity[None, :]
                    .expand_as(prototypes_of_correct_class)
                    .to(device)
                )  # Shape: (num_instances, num_prototypes)

                # Group activations by class
                max_activation_per_class = torch.full(
                    (max_activations.size(0), self.model.num_classes),
                    -float("inf"),
                    device=device,
                )  # Shape: (num_instances, num_classes)
                max_activation_per_class.scatter_reduce_(
                    1,
                    class_indices,
                    correct_class_activations,
                    reduce="amax",
                    include_self=False,
                )

                # Filter activations by relevance and compute the minimum for each instance
                filtered_activations = torch.where(
                    relevant_classes_mask.to(device),
                    max_activation_per_class.gather(1, class_indices),
                    torch.tensor(float("inf"), device=device),
                )  # Shape: (num_instances, num_prototypes)

                # Get the minimum activation of relevant classes for each instance
                min_activation_relevant_classes, _ = torch.min(
                    filtered_activations, dim=1
                )  # Shape: (num_instances,)

                # Compute the cluster cost as the mean of these min activations, explicitly ignoring 'inf' values
                finite_activations = min_activation_relevant_classes[
                    torch.isfinite(min_activation_relevant_classes)
                ]
                cluster_cost = (
                    torch.mean(finite_activations)
                    if len(finite_activations) > 0
                    else torch.tensor(0.0, device=device)
                )
            else:
                # If prototypes of the correct class are not provided
                maximum_max_activations, _ = torch.max(max_activations, dim=1)
                cluster_cost = torch.mean(maximum_max_activations)

        else:
            # Unsupported task case
            raise NotImplementedError(
                f"Task {self.task} is not supported for cluster cost calculation."
            )

        return cluster_cost

    def calculate_separation_cost(
        self, max_activations: torch.Tensor, prototypes_of_wrong_class: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the separation cost based on the maximum activations and the prototypes of the wrong class.

        The separation cost represents how close the activations are to the prototypes of the incorrect class.

        Args:
            max_activations (torch.Tensor): The maximum activations from the model.
            prototypes_of_wrong_class (torch.Tensor): The prototypes of the wrong class.

        Returns:
            torch.Tensor: The computed separation cost.
        """
        incorrect_class_prototype_activations, _ = torch.max(
            max_activations * prototypes_of_wrong_class, dim=1
        )

        separation_cost = torch.mean(incorrect_class_prototype_activations)

        return separation_cost

    def calculate_orthogonality_loss(self) -> torch.Tensor:
        """
        Calculate the normalized orthogonality loss.

        Returns:
            torch.Tensor: The normalized orthogonality loss.
        """
        orthogonalities = self.model.get_prototype_orthogonalities()
        orthogonality_loss = torch.norm(orthogonalities)

        # Normalize the orthogonality loss by the number of elements
        normalized_orthogonality_loss = orthogonality_loss / orthogonalities.numel()

        return normalized_orthogonality_loss

    def calculate_l1_loss_frequency_weights(self) -> torch.Tensor:
        """
        Compute the mean L1 norm of the model's frequency weights for each prototype.

        Returns:
            torch.Tensor: The mean L1 norm.
        """
        if self.model.frequency_weights is None:
            return torch.tensor(0.0)

        # Apply the sigmoid function to the frequency weights
        sigmoid_weights = torch.sigmoid(self.model.frequency_weights)

        # Calculate the L1 norm over all weights
        l1_norm = torch.norm(sigmoid_weights, p=1)

        # Divide the L1 norm by the number of weights
        mean_l1_norm = l1_norm / sigmoid_weights.numel()

        return mean_l1_norm

    def calculate_l1_loss_last_layer(self) -> torch.Tensor:
        """
        Compute the mean L1 norm of the model's last layer weights.

        Returns:
            torch.Tensor: The computed mean L1 norm.
        """
        last_layer = self.model.last_layer

        if isinstance(last_layer, torch.nn.Linear):
            # Calculate the L1 norm over all weights
            l1_norm = torch.norm(last_layer.weight, p=1)
        elif isinstance(last_layer, LinearLayerWithoutNegativeConnections):
            # Apply ReLU to weights if non_negative is True
            weight = (
                torch.relu(last_layer.weight)
                if last_layer.non_negative
                else last_layer.weight
            )
            # Calculate the L1 norm over all weights
            l1_norm = torch.norm(weight, p=1)
        else:
            raise ValueError("Unsupported layer type for L1 loss computation")

        # Divide the L1 norm by the number of weights
        mean_l1_norm = l1_norm / last_layer.weight.numel()

        return mean_l1_norm

    def calculate_zero_weights_loss_last_layer(self) -> torch.Tensor:
        """
        Compute the zero weight loss of the model's last layer weights.

        This function calculates a penalty for having all weights of any class close to zero.
        It uses an exponential decay to provide a smooth, continuous penalty.

        Returns:
            torch.Tensor: The computed zero weight loss.
        """
        last_layer = self.model.last_layer
        weight = last_layer.weight

        if isinstance(last_layer, LinearLayerWithoutNegativeConnections):
            # Apply ReLU to weights if non_negative is True
            weight = torch.relu(weight) if last_layer.non_negative else weight

        # Calculate the minimum over the sum of the absolute weights of each class.
        min_sum_abs_weights = torch.min(torch.sum(torch.abs(weight), dim=1))
        # Apply an exponential function to penalize sums near zero
        zero_weights_loss = torch.exp(-min_sum_abs_weights)

        return zero_weights_loss

    def calculate_total_loss(
        self,
        cross_entropy_loss: torch.Tensor,
        cluster_cost: torch.Tensor,
        separation_cost: torch.Tensor,
        l1_loss_last_layer: torch.Tensor,
        zero_weights_loss_last_layer: torch.Tensor,
        l1_loss_frequency_weights: torch.Tensor,
        orthogonality_loss: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the total loss by combining various components: cross-entropy, cluster cost,
        separation cost, L1 loss, and optionally, orthogonality loss. The weights of these
        components can be set using the `coefficients` dictionary. If not provided, default values are used.

        Args:
            cross_entropy_loss (torch.Tensor): The cross-entropy loss.
            cluster_cost (torch.Tensor): The cluster cost.
            separation_cost (torch.Tensor): The separation cost.
            l1_loss_last_layer (torch.Tensor): The L1 loss for the last layer weights of the model.
            l1_loss_frequency_weights (torch.Tensor): The L1 loss for the frequency weights of the model.
            orthogonality_loss (torch.Tensor): The orthogonality loss.

        Returns:
            torch.Tensor: The computed total loss.
        """

        # Default coefficients
        default_coeffs = {
            "cross_entropy_loss": 1.0,
            "cluster_cost": -0.25,
            "separation_cost": 0.05,
            "l1_loss_last_layer": None,
            "zero_weights_loss_last_layer": None,
            "l1_loss_frequency_weights": None,
            "orthogonality_loss": 1.0,
        }

        # If coefficients are provided, update the defaults
        if self.coefficients is not None:
            default_coeffs.update(self.coefficients)

        # Compute total loss
        loss = default_coeffs["cross_entropy_loss"] * cross_entropy_loss

        # Add cluster cost
        if self.coefficients["cluster_cost"] is not None:
            loss += default_coeffs["cluster_cost"] * cluster_cost

        # Add separation cost
        if self.coefficients["separation_cost"] is not None:
            loss += default_coeffs["separation_cost"] * separation_cost

        # Add orthogonality loss if applicable
        if self.coefficients["orthogonality_loss"] is not None:
            loss += default_coeffs["orthogonality_loss"] * orthogonality_loss

        # Add L1 loss for last layer weights
        if self.coefficients["l1_loss_last_layer"] is not None:
            loss += default_coeffs["l1_loss_last_layer"] * l1_loss_last_layer

        # Add zero weights loss for last layer
        if self.coefficients["zero_weights_loss_last_layer"] is not None:
            loss += (
                default_coeffs["zero_weights_loss_last_layer"]
                * zero_weights_loss_last_layer
            )

        # Add L1 loss for frequency weights if applicable
        if (
            self.model.frequency_weights is not None
            and self.coefficients["l1_loss_frequency_weights"] is not None
        ):
            loss += (
                default_coeffs["l1_loss_frequency_weights"] * l1_loss_frequency_weights
            )

        return loss

    def push_prototypes(
        self,
        dataloader,
        save_prototype_waveform_files: bool,
        save_prototype_spectrogram_files: bool,
    ) -> None:
        """
        This method pushes prototypes through the network using the specified training push data loader and saves
        them to a directory if required. It serves as a wrapper around the `push_prototypes` function from another
        module, encapsulating additional details specific to this class's context.

        The function involves setting up various parameters needed by the `push_prototypes` function, including
        dataloader, network, and configuration settings. It also determines the directory for saving prototype
        files based on the `save_prototype_files` flag.

        Args:
            save_prototype_files (bool): A flag indicating whether to save the prototype files. If True,
                                         prototypes are saved in the directory specified by `self.prototype_files_dir`.

        Returns:
            None: This method does not return anything.
        """

        # Determine the spectrogram dimensions from the first sample
        first_sample = dataloader.dataset[0]
        self.spectrogram_height, self.spectrogram_width = first_sample[
            "input_values"
        ].shape[1:3]

        # Call the push_prototypes function with necessary parameters
        push_prototypes(
            dataloader=dataloader,
            prototype_network=self.model,
            prototype_layer_stride=self.prototype_layer_stride,
            root_dir_for_saving_prototypes=self.prototype_files_dir,
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            label_to_category_mapping=self.label_to_category_mapping,
            mean=self.train_mean,
            std=self.train_std,
            n_mels=self.n_mels,
            spectrogram_height=self.spectrogram_height,
            spectrogram_width=self.spectrogram_width,
            save_prototype_waveform_files=save_prototype_waveform_files,
            save_prototype_spectrogram_files=save_prototype_spectrogram_files,
        )
