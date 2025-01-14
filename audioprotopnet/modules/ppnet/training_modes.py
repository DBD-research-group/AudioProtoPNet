import torch


def apply_gradient_mask(grad: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Apply a mask to the gradients.

    This function is used as a hook in PyTorch to modify the gradients of a layer's weights during the backward pass.
    It performs an element-wise multiplication of the gradients with a mask, effectively zeroing out the gradients
    for certain weights based on the mask.

    Args:
        grad (torch.Tensor): The gradient tensor that needs to be modified. This tensor comes from the backward pass.
        mask (torch.Tensor): The mask tensor that will be applied to the gradient tensor. The mask should be of the
                             same shape as the gradient tensor. Elements in the mask should be 0 where the gradients
                             need to be zeroed out, and 1 where gradients should be kept.

    Returns:
        torch.Tensor: The modified gradient tensor after applying the mask.
    """

    # Ensure the mask is on the same device as the gradient tensor
    mask = mask.to(grad.device)

    # Perform element-wise multiplication of the gradient and the mask
    # This operation will zero out the gradients where the mask has 0 values
    return grad * mask


def last_only(
    model: torch.nn.Module, freeze_incorrect_class_connections: bool = False
) -> None:
    """
    Freeze all layers except specified weights in the last layer for a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model to modify.
        freeze_incorrect_class_connections (bool): If True, only train weights between prototypes and
                                                   their corresponding classes. If False, train all weights
                                                   in the last layer. Defaults to False.

    Returns:
        None
    """
    # Freeze all layers except the last layer
    for p in model.backbone_model.parameters():
        p.requires_grad = False
    for p in model.add_on_layers.parameters():
        p.requires_grad = False
    model.prototype_vectors.requires_grad = False

    if model.frequency_weights is not None:
        model.frequency_weights.requires_grad = False

    if freeze_incorrect_class_connections:
        correct_connections_mask = (
            torch.t(model.prototype_class_identity)
            .float()
            .to(model.last_layer.weight.device)
        )
        mask = correct_connections_mask

        # Remove existing hooks
        if hasattr(model.last_layer, "hooks"):
            for hook in model.last_layer.hooks:
                hook.remove()
            del model.last_layer.hooks

        # Attach a new hook to apply the gradient mask
        hook = model.last_layer.weight.register_hook(
            lambda grad: apply_gradient_mask(grad, mask)
        )
        model.last_layer.hooks = [hook]
    else:
        # Train all weights in the last layer
        for p in model.last_layer.parameters():
            p.requires_grad = True


def warm_only(model: torch.nn.Module, last_layer_fixed: bool) -> None:
    """Freeze all layers except for the add-on layers and prototype vectors for a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model to modify.
        last_layer_fixed (bool): If True, the last layer's gradients are disabled (frozen).
                                 If False, the last layer's gradients are enabled (trainable).

    Returns:
        None
    """
    # Freeze all layers except for the add-on layers and prototype vectors
    for p in model.backbone_model.parameters():
        p.requires_grad = False
    for p in model.add_on_layers.parameters():
        p.requires_grad = True
    model.prototype_vectors.requires_grad = True

    # Set requires_grad for the last layer based on last_layer_fixed
    for p in model.last_layer.parameters():
        p.requires_grad = not last_layer_fixed

    if model.frequency_weights is not None:
        model.frequency_weights.requires_grad = True


def joint(
    model: torch.nn.Module,
    last_layer_fixed: bool = True,
    freeze_incorrect_class_connections: bool = False,
) -> None:
    """Unfreeze all layers except the last layer for a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model to modify.
        last_layer_fixed (bool): Whether to fix the parameters of the last layer (default: True).
        freeze_incorrect_class_connections (bool): If True, only train weights between prototypes and
                                                   their corresponding classes. If False, train all weights
                                                   in the last layer. Defaults to False.

    Returns:
        None
    """
    # Unfreeze all layers except the last layer
    for p in model.backbone_model.parameters():
        p.requires_grad = True
    for p in model.add_on_layers.parameters():
        p.requires_grad = True
    model.prototype_vectors.requires_grad = True

    if model.frequency_weights is not None:
        model.frequency_weights.requires_grad = True

    if last_layer_fixed:
        for p in model.last_layer.parameters():
            p.requires_grad = False
    else:
        if freeze_incorrect_class_connections:
            correct_connections_mask = (
                torch.t(model.prototype_class_identity)
                .float()
                .to(model.last_layer.weight.device)
            )
            mask = correct_connections_mask

            # Remove existing hooks
            if hasattr(model.last_layer, "hooks"):
                for hook in model.last_layer.hooks:
                    hook.remove()
                del model.last_layer.hooks

            # Attach a new hook to apply the gradient mask
            hook = model.last_layer.weight.register_hook(
                lambda grad: apply_gradient_mask(grad, mask)
            )
            model.last_layer.hooks = [hook]
        else:
            # Train all weights in the last layer
            for p in model.last_layer.parameters():
                p.requires_grad = True


def freeze_all_layers(model: torch.nn.Module) -> None:
    """Freeze all layers for a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model to modify.

    Returns:
        None
    """
    # Freeze all layers
    for p in model.backbone_model.parameters():
        p.requires_grad = False
    for p in model.add_on_layers.parameters():
        p.requires_grad = False
    model.prototype_vectors.requires_grad = False
    for p in model.last_layer.parameters():
        p.requires_grad = False

    if model.frequency_weights is not None:
        model.frequency_weights.requires_grad = False
