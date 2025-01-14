import json
import os
from typing import Dict, List, Optional

from birdset import utils
import datasets
import hydra
import lightning as L
from lightning import LightningModule, LightningDataModule
import numpy as np
from omegaconf import OmegaConf
import pyrootutils

from audioprotopnet.modules.checkpoint_loading import load_state_dict
from audioprotopnet.modules.ppnet.lightningprotopnet import LightningProtoPNet


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


def initialize_model(
    config: Dict[str, any],
    label_counts: int,
    label_to_category_mapping: Dict[int, str],
    train_batch_size: int,
    train_mean: np.ndarray,
    train_std: np.ndarray,
    len_trainset: int,
    start_phase: Optional[str],
    ckpt: Optional[str],
) -> LightningProtoPNet:
    """
    Initialize the model based on the configuration settings.

    Args:
        config (Dict[str, any]): A dictionary containing configuration settings.
        logger (LightningLogger): Logger for the training process.
        image_height (int): The height of the input images.
        image_width (int): The width of the input images.
        train_mean (np.ndarray): The mean of the training data, used for normalization.
        train_std (np.ndarray): The standard deviation of the training data, used for normalization.

    Returns:
        LightningProtoPNet: The initialized model.
    """

    # Setup directory for saving prototypes and related files
    prototype_files_dir = os.path.join(config.paths.output_dir, "prototype_files")
    os.makedirs(prototype_files_dir, exist_ok=True)

    learning_rates = {
        "warm": config.module.warm_optimizer_lrs,
        "joint": config.module.joint_optimizer_lrs,
        "last_layer": config.module.last_layer_optimizer_lr,
    }

    # Determine the penultimate level of model training.
    if (
        not config.module.last_layer_fixed
        and config.module.pruning_threshold is not None
    ):
        penultimate_phase = "pruning"
    elif not config.module.last_layer_fixed:
        penultimate_phase = "joint_with_last_layer"
    else:
        penultimate_phase = "joint"

    # Conditional settings based on the training phase
    phase_settings = {
        "warm": {
            "init_weights": True,
            "checkpoint_suffix": None,
            "num_epochs": config.module.max_epochs_warm,
        },
        "joint": {
            "init_weights": False,
            "checkpoint_suffix": "warm",
            "num_epochs": config.module.max_epochs_joint,
        },
        "joint_with_last_layer": {
            "init_weights": False,
            "checkpoint_suffix": "warm",
            "num_epochs": config.module.max_epochs_joint_with_last_layer,
        },
        "push_joint_with_last_layer": {
            "init_weights": False,
            "checkpoint_suffix": "joint_with_last_layer",
            "num_epochs": None,
        },
        "last_layer": {
            "init_weights": False,
            "checkpoint_suffix": (
                "push_joint_with_last_layer"
                if config.module.pruning_threshold is not None
                else "push_final"
            ),
            "num_epochs": config.module.max_epochs_last_layer,
        },
        "pruning": {
            "init_weights": False,
            "checkpoint_suffix": "last_layer",
            "num_epochs": None,
        },
        "push_final": {
            "init_weights": False,
            "checkpoint_suffix": penultimate_phase,
            "num_epochs": None,
        },
    }

    current_phase = config.module.training_phase

    (
        init_weights,
        checkpoint_suffix,
        num_epochs,
    ) = (
        phase_settings[current_phase]["init_weights"],
        phase_settings[current_phase]["checkpoint_suffix"],
        phase_settings[current_phase]["num_epochs"],
    )

    checkpoint = (
        None
        if current_phase == "warm"
        or (
            start_phase is not None
            and ckpt is not None
            and current_phase == start_phase
        )
        else f"{config.callbacks.model_checkpoint.dirpath}/{config.module.network.model_name}_{checkpoint_suffix}.ckpt"
    )

    prototype_shape = config.module.network.model.prototype_shape

    # Setup model
    log.info(f"Instantiate PPNet model <{config.module.network.model._target_}>")
    ppnet = hydra.utils.instantiate(
        config.module.network.model,
        prototype_shape=prototype_shape,
        init_weights=init_weights,
    )

    if checkpoint:
        state_dict = load_state_dict(checkpoint)
        ppnet.load_state_dict(state_dict, strict=True)

    network_dict = {
        "model": ppnet,
        "model_name": config.module.network.model_name,
        "model_type": config.module.network.model_type,
        "torch_compile": config.module.network.torch_compile,
    }

    model_args = {
        "network": network_dict,
        "output_activation": config.module.output_activation,
        "loss": config.module.loss,
        "optimizer": config.module.optimizer,
        "lr_scheduler": config.module.lr_scheduler,
        "metrics": config.module.metrics,
        "logging_params": config.module.logging_params,
        "len_trainset": len_trainset,
        "num_epochs": num_epochs,
        "batch_size": train_batch_size,
        "label_counts": label_counts,
        "task": config.module.task,
        "class_weights_loss": config.module.class_weights_loss,
        "num_gpus": config.module.num_gpus,
        "prediction_table": config.module.prediction_table,
        "training_phase": config.module.training_phase,
        "learning_rates": learning_rates,
        "prototype_files_dir": prototype_files_dir,
        "label_to_category_mapping": label_to_category_mapping,
        "train_mean": train_mean,
        "train_std": train_std,
        "last_layer_fixed": config.module.last_layer_fixed,
        "freeze_incorrect_class_connections": config.module.freeze_incorrect_class_connections,
        "subtractive_margin": config.module.subtractive_margin,
        "coefficients": config.module.coefs,
        "weight_decay": config.module.weight_decay,
        "prototype_layer_stride": config.module.prototype_layer_stride,
        "sample_rate": config.module.network.sampling_rate,
        "n_fft": config.datamodule.transforms.preprocessing.spectrogram_conversion.n_fft,
        "hop_length": config.datamodule.transforms.preprocessing.spectrogram_conversion.hop_length,
        "n_mels": config.datamodule.transforms.preprocessing.melscale_conversion.n_mels,
    }

    model = LightningProtoPNet(**model_args)

    # if current_phase == "joint_with_last_layer" and model.model.incorrect_class_connection:
    #     # Set the incorrect class connections in the last layer to zero to avoid unnecessary optimization steps that do
    #     # just that.
    #     model.model.set_last_layer_incorrect_connection(incorrect_strength=0.0)

    return model


def run_training_phase(
    config: Dict[str, any],
    callbacks,
    model: LightningModule,
    datamodule: LightningDataModule,
    ckpt_path: Optional[str],
    logger,
) -> None:
    """
    Run a specific training phase for the model.

    Args:
        config (Dict[str, any]): Configuration settings containing training parameters.
        lightning_model (LightningProtoPNet): The model to be trained.
        logger (LightningLogger): Logger to track training progress and metrics.

    This function sets up the PyTorch Lightning Trainer and runs the training process
    for the provided model using the specified data loaders. It utilizes the configuration
    settings for setting up the training parameters like batch size, learning rate, etc.
    """

    log.info("Instantiate trainer <{config.trainer._target_}>")
    trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, max_epochs=model.num_epochs
    )

    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=ckpt_path,
    )


def run_push_phase(
    config: Dict[str, any],
    callbacks,
    model: LightningProtoPNet,
    datamodule: LightningDataModule,
    dataloader,
    save_prototype_waveform_files: bool,
    save_prototype_spectrogram_files: bool,
    checkpoint_suffix: str,
    logger,
) -> None:
    """
    Executes the push phase in the training process of a prototypical network model.

    This function triggers the push operation on prototypes, validates the model
    using the test dataset, and saves the model checkpoint after the push phase.

    Args:
        config (Dict[str, any]): Configuration settings containing training parameters.
        lightning_model (LightningProtoPNet): The model to be trained.
        save_prototype_files (bool): Flag to determine whether to save prototype files during push operation.
        checkpoint_suffix (str): Suffix to append to the checkpoint file name.
        logger (LightningLogger): Logger to track training progress and metrics.

    Returns:
        Tuple[float, float, float, float]: The validation loss, validation accuracy, test loss, and test accuracy
    """

    # Push prototypes
    model.push_prototypes(
        dataloader=dataloader,
        save_prototype_waveform_files=save_prototype_waveform_files,
        save_prototype_spectrogram_files=save_prototype_spectrogram_files,
    )

    # Set up the PyTorch Lightning Trainer
    trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger
    )

    # Validate the model after the push operation (required to save the push phase checkpoint)
    trainer.validate(model=model, datamodule=datamodule)

    # Save the model state after the push operation.
    trainer.save_checkpoint(
        f"{config.callbacks.model_checkpoint.dirpath}/{config.module.network.model_name}_{checkpoint_suffix}.ckpt"
    )


def get_training_phases(
    train_classifier_only: bool,
    last_layer_fixed: bool,
    pruning_threshold: Optional[float] = None,
    start_phase: Optional[str] = None,
) -> List[str]:
    """
    Sequentially execute training phases.

    This function determines the sequence of training phases based on whether
    the last layer is fixed and optionally starts from a specified phase.

    Args:
        train_classifier_only (bool): If True, only the warm phase is used.
        last_layer_fixed (bool): Indicates if the last layer is fixed.
        start_phase (Optional[str]): The phase from which to start. If None,
                                     all phases are returned in the default order.

    Returns:
        List[str]: The ordered list of training phases.

    Raises:
        ValueError: If the provided start_phase is not valid.
    """

    # If train_classifier_only is True, return only the "warm" phase
    if train_classifier_only:
        return ["warm"]

    # Determine the training phases based on whether the last layer is fixed
    if not last_layer_fixed and pruning_threshold is not None:
        phases = [
            "warm",
            "joint_with_last_layer",
            "push_joint_with_last_layer",
            "last_layer",
            "pruning",
            "push_final",
        ]
    elif not last_layer_fixed:
        phases = [
            "warm",
            "joint_with_last_layer",
            "push_final",
            "last_layer",
        ]
    else:
        phases = ["warm", "joint", "push_final"]

    # If a start phase is specified, slice the phases list to start from the specified phase
    if start_phase is not None:
        if start_phase in phases:
            start_index = phases.index(start_phase)
            phases = phases[
                start_index:
            ]  # Slice the phases list to start from the start_phase
        else:
            raise ValueError(f"Invalid start_phase: {start_phase}")

    return phases


def run_pruning_phase(
    config: Dict[str, any],
    callbacks,
    model: LightningProtoPNet,
    datamodule: LightningDataModule,
    logger,
) -> LightningProtoPNet:
    """
    Executes the pruning phase in the training process of a prototypical network model.

    Args:
        config (Dict[str, any]): Configuration settings containing training parameters.
        lightning_model (LightningProtoPNet): The model to be trained.
        logger (LightningLogger): Logger to track training progress and metrics.

    Returns:
        LightningProtoPNet: The pruned ProtoPNet model.
    """

    # Prune all prototypes whose connection to their corresponding class is below a certain threshold.
    model.model.prune_prototypes_by_threshold(threshold=config.module.pruning_threshold)

    log.info(
        f"New number of prototypes after pruning: {model.model.num_prototypes_after_pruning}"
    )

    # Set up the PyTorch Lightning Trainer
    trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger
    )

    # Validate the model after the push operation (required to save the pruning phase checkpoint)
    trainer.validate(model=model, datamodule=datamodule)

    # Save the model state after the pruning operation.
    trainer.save_checkpoint(
        f"{config.callbacks.model_checkpoint.dirpath}/{config.module.network.model_name}_pruning.ckpt"
    )

    return model


@utils.register_custom_resolvers(**_HYDRA_PARAMS)
@hydra.main(**_HYDRA_PARAMS)
def main_audioprotopnet(cfg):
    log.info("Using config: \n%s", OmegaConf.to_yaml(cfg))

    log.info(f"Dataset path: <{os.path.abspath(cfg.paths.dataset_path)}>")
    os.makedirs(cfg.paths.dataset_path, exist_ok=True)

    log.info(f"Log path: <{os.path.abspath(cfg.paths.log_dir)}>")
    os.makedirs(cfg.paths.log_dir, exist_ok=True)

    log.info(f"Root Dir:<{os.path.abspath(cfg.paths.log_dir)}>")
    log.info(f"Work Dir:<{os.path.abspath(cfg.paths.work_dir)}>")
    log.info(f"Output Dir:<{os.path.abspath(cfg.paths.output_dir)}>")
    log.info(f"Background Dir:<{os.path.abspath(cfg.paths.background_path)}>")

    log.info(f"Seed everything with <{cfg.seed}>")
    L.seed_everything(cfg.seed)

    # Setup data
    log.info(f"Instantiate datamodule <{cfg.datamodule._target_}>")
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.prepare_data()  # has to be called before model for len_traindataset!

    log.info(f"Instantiate push datamodule <{cfg.datamodule_push._target_}>")
    datamodule_push = hydra.utils.instantiate(cfg.datamodule_push)
    datamodule_push.prepare_data()  # has to be called before model for len_traindataset!
    datamodule_push.setup(stage="fit")
    train_push_loader = datamodule_push.train_dataloader()

    if cfg.datamodule.dataset.dataset_name == "esc50":
        label_to_category_mapping = datamodule.label_to_category_mapping
    else:
        ebird_codes_list = datasets.load_dataset_builder(
            cfg.datamodule.dataset.hf_path, cfg.datamodule.dataset.hf_name
        ).info.features["ebird_code"]
        label_to_category_mapping = dict(enumerate(ebird_codes_list.names))

    # Setup logger
    log.info("Instantiate logger")
    logger = utils.instantiate_loggers(cfg.get("logger"))

    if cfg.get("train"):
        log.info("Starting training")

        start_phase = cfg.get("start_phase")

        training_phases = get_training_phases(
            train_classifier_only=cfg.module.train_classifier_only,
            last_layer_fixed=cfg.module.last_layer_fixed,
            pruning_threshold=cfg.module.pruning_threshold,
            start_phase=start_phase,
        )

        for phase in training_phases:
            log.info(f"Running {phase} training phase")

            if start_phase is not None and phase == start_phase:
                ckpt = cfg.get("ckpt_path")
                if ckpt:
                    log.info(f"Resume training from checkpoint {ckpt}")
                else:
                    log.info("No checkpoint found. Training from scratch!")
            else:
                ckpt = None
                log.info("Training from scratch!")

            cfg.module.training_phase = phase

            # Setup callbacks
            log.info("Instantiate callbacks")
            callbacks = utils.instantiate_callbacks(cfg["callbacks"])

            # Set up model for the current phase
            model = initialize_model(
                config=cfg,
                train_batch_size=datamodule.loaders_config.train.batch_size,
                train_mean=datamodule.transforms.preprocessing.mean,
                train_std=datamodule.transforms.preprocessing.std,
                len_trainset=datamodule.len_trainset,
                label_counts=datamodule.num_train_labels,
                label_to_category_mapping=label_to_category_mapping,
                start_phase=start_phase,
                ckpt=ckpt,
            )

            if phase in (
                "push_joint_with_last_layer",
                "push_final",
            ):
                save_prototype_waveform_files = bool(
                    phase == "push_final" and cfg.save_prototype_waveform_files
                )
                save_prototype_spectrogram_files = bool(
                    phase == "push_final" and cfg.save_prototype_spectrogram_files
                )

                # Run push phase
                run_push_phase(
                    config=cfg,
                    callbacks=callbacks,
                    model=model,
                    datamodule=datamodule,
                    dataloader=train_push_loader,
                    save_prototype_waveform_files=save_prototype_waveform_files,
                    save_prototype_spectrogram_files=save_prototype_spectrogram_files,
                    checkpoint_suffix=phase,
                    logger=logger,
                )

            elif phase == "pruning":
                # Run pruning phase
                run_pruning_phase(
                    config=cfg,
                    callbacks=callbacks,
                    model=model,
                    datamodule=datamodule,
                    logger=logger,
                )

            else:
                # Run training phase
                run_training_phase(
                    config=cfg,
                    callbacks=callbacks,
                    model=model,
                    datamodule=datamodule,
                    ckpt_path=ckpt,
                    logger=logger,
                )

    # Set up the PyTorch Lightning Trainer
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("No ckpt saved or found. Using current weights for testing")
            ckpt_path = None
        else:
            log.info(
                f"The best checkpoint for {cfg.callbacks.model_checkpoint.monitor}"
                f" is {trainer.checkpoint_callback.best_model_score}"
                f" and saved in {ckpt_path}"
            )
        trainer.test(
            model=model,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
        )

    test_metrics = trainer.callback_metrics

    if cfg.get("save_state_dict"):
        log.info("Saving state dicts")
        utils.save_state_dicts(
            trainer=trainer,
            model=model,
            dirname=cfg.paths.output_dir,
            **cfg.extras.state_dict_saving_params,
        )

    if cfg.get("dump_metrics"):
        log.info(f"Dumping final metrics locally to {cfg.paths.output_dir}")
        metric_dict = {**train_metrics, **test_metrics}

        metric_dict = [
            {"name": k, "value": v.item() if hasattr(v, "item") else v}
            for k, v in metric_dict.items()
        ]

        metric_dict.append({"num_prototypes": model.model.num_prototypes})

        file_path = os.path.join(cfg.paths.output_dir, "finalmetrics.json")
        with open(file_path, "w") as json_file:
            json.dump(metric_dict, json_file)

    utils.close_loggers()


if __name__ == "__main__":
    main_audioprotopnet()
