from typing import Dict, Optional, Union

import torch


def load_state_dict(
    checkpoint_path: str, map_location: Optional[Union[str, torch.device]] = None
) -> Dict[str, torch.Tensor]:
    """
    Load a model checkpoint and return the state dictionary. The function is compatible with
    checkpoints from both plain PyTorch and PyTorch Lightning.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        map_location (Optional[Union[str, torch.device]]): Specifies how to remap storage locations
        (for CPU/GPU compatibility). It can be a string ('cpu', 'cuda'), or a torch.device object.

    Returns:
        Dict[str, torch.Tensor]: The state dictionary extracted from the checkpoint. The keys are layer names,
        and values are parameter tensors of the model.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist at the given path.
        IOError: If the checkpoint file cannot be loaded for reasons other than non-existence.
    """
    # Load the checkpoint file. torch.load() automatically loads the tensor to the specified device
    # if map_location is provided.
    print(">> Found a checkpoint. Loading..")
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    # Extract the state dictionary from the checkpoint. The structure of the checkpoint can vary
    # depending on whether it's from plain PyTorch or PyTorch Lightning.
    if "state_dict" in checkpoint:
        # PyTorch Lightning checkpoints usually nest the model state dictionary under the 'state_dict' key.
        state_dict = checkpoint["state_dict"]
    else:
        # For plain PyTorch checkpoints, the file directly contains the state dictionary.
        state_dict = checkpoint

    # Update this part to handle the necessary key replacements
    adjusted_state_dict = {}
    for key, value in state_dict.items():
        # Handle 'model.model.' prefix
        new_key = key.replace("model.model.", "")

        # Handle 'model._orig_mod.model.' prefix
        new_key = new_key.replace("model._orig_mod.model.", "")

        # Handle 'model._orig_mod.' prefix
        new_key = new_key.replace("model._orig_mod.", "")

        new_key = new_key.replace(
            "backbone_embeddings", "backbone_model.model.embeddings"
        )
        new_key = new_key.replace("backbone_encoder", "backbone_model.model.encoder")
        new_key = new_key.replace(
            "backbone_layernorm", "backbone_model.model.layernorm"
        )

        # Map 'backbone_features' to 'backbone_model.model.features'
        new_key = new_key.replace("backbone_features", "backbone_model.model.features")

        # Handle 'model._orig_mod.' prefix
        new_key = new_key.replace(
            "model.backbone_model.model.", "backbone_model.model."
        )
        new_key = new_key.replace("model.prototype_vectors", "prototype_vectors")
        new_key = new_key.replace("model.last_layer.weight", "last_layer.weight")
        new_key = new_key.replace("model.last_layer.bias", "last_layer.bias")
        new_key = new_key.replace("model.frequency_weights", "frequency_weights")

        # Assign the adjusted key
        adjusted_state_dict[new_key] = value

    return adjusted_state_dict
