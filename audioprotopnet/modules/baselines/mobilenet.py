import logging
from typing import Dict, Optional, Literal, Union

import datasets
import torch
from torch import nn
from torchvision.models import mobilenet_v2, mobilenet_v3_large, mobilenet_v3_small

from audioprotopnet.backbones.mobilenet.model import get_model
from audioprotopnet.backbones.mobilenet.utils import name_to_width
from audioprotopnet.helpers import load_state_dict
from audioprotopnet.modules.baselines.efficientnet import (
    update_first_cnn_layer,
)


# MobileNet versions available here: https://github.com/fschmid56/EfficientAT/tree/main
# The naming convention for the models is <model><width_mult>_<dataset>.
# In this sense, mn10_as defines a MobileNetV3 with parameter width_mult=1.0, pre-trained on AudioSet.
# All models available are pre-trained on ImageNet by default (otherwise denoted as 'no_im_pre'),
# followed by training on AudioSet.
MobileNetVersion = Literal[
    "mn10_im_pytorch",
    "mn01_im",
    "mn02_im",
    "mn04_im",
    "mn05_im",
    "mn10_im",
    "mn20_im",
    "mn30_im",
    "mn40_im",
    "mn01_as",
    "mn02_as",
    "mn04_as",
    "mn05_as",
    "mn10_as",
    "mn20_as",
    "mn30_as",
    "mn40_as",
    "mn40_as(2)",
    "mn40_as(3)",
    "mn40_as_no_im_pre",
    "mn40_as_no_im_pre(2)",
    "mn40_as_no_im_pre(3)",
    "mn40_as_ext",
    "mn40_as_ext(2)",
    "mn40_as_ext(3)",
    "mn10_as_hop_5",
    "mn10_as_hop_15",
    "mn10_as_hop_20",
    "mn10_as_hop_25",
    "mn10_as_mels_40",
    "mn10_as_mels_64",
    "mn10_as_mels_256",
    "mn10_as_fc",
    "mn10_as_fc_s2221",
    "mn10_as_fc_s2211",
]

MobileNetVersionTorch = Literal[
    "mobilenet_v2",
    "mobilenet_v3_small",
    "mobilenet_v3_large",
]


class MobileNet(nn.Module):
    """
    MobileNet model for audio classification.

    Attributes:
        architecture (MobileNetVersion): Specifies the MobileNet version to use.
        num_classes (int): Defines the number of classes for the model's output layer.
        num_channels (int): Specifies the number of input channels. Currently, only a
                            single-channel input is supported.
        backbone_mode (bool): If True, the model functions as a feature extractor, and
                              the classifier is replaced with an identity function. This
                              is useful for tasks where only features are needed.
        checkpoint (Optional[str]): Path to a pre-trained model checkpoint for loading
                                    weights. If None, the model will be initialized
                                    without pre-trained weights.
        freeze_backbone (bool): If True, the backbone parameters (weights and biases)
                                are frozen, preventing them from being updated during
                                training.

    """

    def __init__(
        self,
        architecture: Union[MobileNetVersion, MobileNetVersionTorch],
        num_classes: int,
        num_channels: int = 1,
        backbone_mode: bool = False,
        checkpoint: Optional[str] = None,
        freeze_backbone: bool = False,
        audioset_pretrained: bool = False,
        pretrain_info: Optional[Dict] = None,
    ):
        """
        Initialize the MobileNet model.

        Args:
            architecture (Union[MobileNetVersion, MobileNetVersionTorch]): The MobileNet architecture version.
            num_classes (int): Number of output classes for classification.
            num_channels (int): Number of input channels (default is 1).
            backbone_mode (bool): Enables backbone mode, turning the model into a feature
                                  extractor (default is False).
            checkpoint (Optional[str]): Path to the checkpoint file for initializing model
                                        weights (default is None).
            freeze_backbone (bool): Indicates whether the backbone should be frozen during
                                    training (default is False).
            audioset_pretrained (bool): Whether to initialize with weights pretrained on AudioSet. Default is False.
        """
        super().__init__()

        self.architecture = architecture
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.checkpoint = checkpoint
        self.backbone_mode = backbone_mode
        self.freeze_backbone = freeze_backbone
        self.audioset_pretrained = audioset_pretrained

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

        self.model = None

        self._initialize_model()

        # Optionally freeze the backbone
        if self.freeze_backbone:
            self._freeze_backbone()

    def _initialize_model(self) -> nn.Module:
        """Initializes the MobileNet model based on specified attributes.

        Returns:
            nn.Module: The initialized MobileNet model.
        """

        # Error handling for incompatible audioset_pretrained settings
        if self.audioset_pretrained and (
            self.architecture
            in ["mobilenet_v2", "mobilenet_v3_large", "mobilenet_v3_small"]
            or self.checkpoint
        ):
            raise ValueError(
                "audioset_pretrained cannot be True if architecture is one of the PyTorch MobileNet versions or a checkpoint is provided."
            )

        # Initialize model based on the backbone architecture
        if self.architecture == "mobilenet_v2":
            mobilenet_model = mobilenet_v2(
                pretrained=False, num_classes=self.num_classes
            )
        elif self.architecture == "mobilenet_v3_large":
            mobilenet_model = mobilenet_v3_large(
                pretrained=False, num_classes=self.num_classes
            )
        elif self.architecture == "mobilenet_v3_small":
            mobilenet_model = mobilenet_v3_small(
                pretrained=False, num_classes=self.num_classes
            )
        else:
            mobilenet_model = get_model(
                num_classes=self.num_classes,
                pretrained_name=self.architecture,
                width_mult=name_to_width(self.architecture),
            )

        # Update the first layer to match num_channels if needed
        update_first_cnn_layer(model=mobilenet_model, num_channels=self.num_channels)

        if self.backbone_mode:
            mobilenet_model.classifier = torch.nn.Identity()
            if self.architecture not in ("mobilenet_v2",):
                mobilenet_model.avgpool = torch.nn.Identity()

        # Load checkpoint if provided
        if self.checkpoint:
            state_dict = load_state_dict(self.checkpoint)
            try:
                mobilenet_model.load_state_dict(state_dict, strict=True)
            except RuntimeError as error:
                logging.info(str(error))
                logging.info("Loading pre-trained weights in a non-strict manner.")
                mobilenet_model.load_state_dict(state_dict, strict=False)

        self.model = mobilenet_model

    def _freeze_backbone(self):
        """
        Freezes the backbone part of the model by setting requires_grad to False
        for all parameters in the backbone.
        """
        for name, child in self.model.named_children():
            if name in ["features"]:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(
        self, input_values: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Defines the forward pass of the MobileNet model.

        Args:
            input_values (torch.Tensor): An input batch.
            labels (Optional[torch.Tensor]): The corresponding labels. Default is None.

        Returns:
            torch.Tensor: The output of the MobileNet model.
        """
        if self.architecture in [
            "mobilenet_v2",
            "mobilenet_v3_large",
            "mobilenet_v3_small",
        ]:
            # Directly return the output for PyTorch MobileNet models
            output = self.model(input_values)
        else:
            # If using a different architecture (handled by the get_model function), assume it's the original setup
            output = self.model(input_values)[0]

        return output
