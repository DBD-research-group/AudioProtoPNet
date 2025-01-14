from typing import Dict, Optional

import datasets
import torch
from torch import nn
from transformers import (
    AutoConfig,
    EfficientNetConfig,
    EfficientNetForImageClassification,
    EfficientNetModel,
)


class EfficientNetClassifier(nn.Module):
    """
    EfficientNet model for audio classification.

    Attributes:
        architecture (EfficientNetVersion): The version of EfficientNet to use.
        num_classes (int): The number of classes for the output layer.
        num_channels (int): The number of input channels.
        checkpoint (Optional[str]): Path to a checkpoint for loading pre-trained weights.
    """

    def __init__(
        self,
        num_classes: int,
        num_channels: int = 1,
        embedding_size: Optional[int] = None,
        backbone_mode: bool = False,
        checkpoint: Optional[str] = None,
        local_checkpoint: Optional[str] = None,
        cache_dir: Optional[str] = None,
        pretrain_info: Optional[Dict] = None,
    ):
        """
        Initialize the EfficientNet model.

        Args:
        num_classes (int): The number of classes for classification.
        num_channels (int): The number of input channels. Default is 1.
        checkpoint (Optional[str]): Path to a checkpoint for loading pre-trained weights. Default is None.
        """
        super().__init__()

        if pretrain_info:
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
            self.hf_path = None
            self.hf_name = None
            self.num_classes = num_classes

        self.num_channels = num_channels
        self.embedding_size = embedding_size
        self.backbone_mode = backbone_mode
        self.checkpoint = checkpoint
        self.local_checkpoint = local_checkpoint
        self.cache_dir = cache_dir

        self.model = None
        self.architecture = None

        self._initialize_model()

    def _initialize_model(self) -> nn.Module:
        """Initializes the EfficientNet model based on specified attributes.

        Returns:
            nn.Module: The initialized EfficientNet model.
        """

        adjusted_state_dict = None

        if self.backbone_mode:
            model = EfficientNetModel
        else:
            model = EfficientNetForImageClassification

        if self.checkpoint:
            if self.local_checkpoint:
                state_dict = torch.load(self.local_checkpoint)["state_dict"]

                # Update this part to handle the necessary key replacements
                adjusted_state_dict = {}
                for key, value in state_dict.items():
                    # Handle 'model.model.' prefix
                    new_key = key.replace("model.model.", "")

                    # Handle 'model._orig_mod.model.' prefix
                    new_key = new_key.replace("model._orig_mod.model.", "")

                    # Assign the adjusted key
                    adjusted_state_dict[new_key] = value

            config = EfficientNetConfig.from_pretrained(self.checkpoint)
            hidden_dim = config.hidden_dim

            if self.embedding_size is not None:
                # Update the last hidden size
                hidden_dim = self.embedding_size

            self.model = model.from_pretrained(
                self.checkpoint,
                num_labels=self.num_classes,
                num_channels=self.num_channels,
                hidden_dim=hidden_dim,
                cache_dir=self.cache_dir,
                state_dict=adjusted_state_dict,
                ignore_mismatched_sizes=True,
            )
        else:
            config = AutoConfig.from_pretrained(
                "google/efficientnet-b1",
                num_labels=self.num_classes,
                num_channels=self.num_channels,
            )
            self.model = model(config)

        self.architecture = self.model.config.model_type

    def forward(
        self, input_values: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Defines the forward pass of the EfficientNet model.

        Args:
            input_values (torch.Tensor): An input batch.
            labels (Optional[torch.Tensor]): The corresponding labels. Default is None.

        Returns:
            torch.Tensor: The output of the EfficientNet model.
        """
        outputs = self.model(input_values)

        if self.backbone_mode:
            output = outputs.last_hidden_state
        else:
            output = outputs.logits

        return output

    @torch.inference_mode()
    def get_logits(self, dataloader, device):
        pass

    @torch.inference_mode()
    def get_probas(self, dataloader, device):
        pass

    @torch.inference_mode()
    def get_representations(self, dataloader, device):
        pass
