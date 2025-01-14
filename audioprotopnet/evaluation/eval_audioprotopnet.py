import json
import os
from typing import Dict

from birdset import utils
import hydra
import lightning as L
import pyrootutils

from audioprotopnet.modules.checkpoint_loading import load_state_dict
from audioprotopnet.modules.ppnet.lightningprotopnet import LightningProtoPNet

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

log = utils.get_pylogger(__name__)


def initialize_inference_model(
    config: Dict[str, any],
    label_counts: int,
    train_batch_size: int,
    len_trainset: int,
    checkpoint: str,
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
        lightning_model_pruned (Optional[LightningProtoPNet]): The PyTorch Lightning ProtoPNet model obtained from the
        pruning phase.

    Returns:
        LightningProtoPNet: The initialized model.
    """
    prototype_shape = config.module.network.model.prototype_shape

    # Setup model
    log.info(f"Instantiate PPNet model <{config.module.network.model._target_}>")
    ppnet = hydra.utils.instantiate(
        config.module.network.model,
        prototype_shape=prototype_shape,
        init_weights=False,
        non_negative_last_layer=None,
    )

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
        "optimizer": None,  # config.module.optimizer,
        "lr_scheduler": None,  # config.module.lr_scheduler,
        "metrics": config.module.metrics,
        "logging_params": config.module.logging_params,
        "len_trainset": len_trainset,
        "num_epochs": None,  # num_epochs,
        "batch_size": train_batch_size,
        "label_counts": label_counts,
        "task": config.module.task,
        "class_weights_loss": config.module.class_weights_loss,
        "num_gpus": config.module.num_gpus,
        "prediction_table": config.module.prediction_table,
        "training_phase": "inference",
        "learning_rates": None,
        "prototype_files_dir": None,
        "label_to_category_mapping": None,
        "train_mean": None,
        "train_std": None,
        "last_layer_fixed": None,
        "freeze_incorrect_class_connections": None,
        "subtractive_margin": None,
        "coefficients": config.module.coefs,
        "weight_decay": None,
        "prototype_layer_stride": config.module.prototype_layer_stride,
        "sample_rate": config.module.network.sampling_rate,
        "n_fft": config.datamodule.transforms.preprocessing.spectrogram_conversion.n_fft,
        "hop_length": config.datamodule.transforms.preprocessing.spectrogram_conversion.hop_length,
        "n_mels": config.datamodule.transforms.preprocessing.melscale_conversion.n_mels,
    }

    model = LightningProtoPNet(**model_args)

    return model


@hydra.main(**_HYDRA_PARAMS)
def eval(cfg):
    log.info("Starting Evaluation")
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

    log.info(f"Instantiate datamodule <{cfg.datamodule._target_}>")
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.prepare_data()

    log.info(f"Instantiate model <{cfg.module.network.model._target_}>")
    # Set up model for the current phase
    model = initialize_inference_model(
        config=cfg,
        train_batch_size=datamodule.loaders_config.train.batch_size,
        len_trainset=datamodule.len_trainset,
        label_counts=datamodule.num_train_labels,
        checkpoint=cfg.ckpt_path,
    )

    # # Attention! If you want to use pruning, you must specify the checkpoint from the last_layer training phase
    # # as the checkpoint for evaluation!
    # if not cfg.module.last_layer_fixed and cfg.module.pruning_threshold is not None:
    #     # Prune all prototypes whose connection to their corresponding class is below a certain threshold.
    #     model.model.prune_prototypes_by_threshold(
    #         threshold=cfg.module.pruning_threshold
    #     )
    #
    #     log.info(
    #         f"New number of prototypes after pruning: {model.model.num_prototypes}"
    #     )

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

    if cfg.get("test"):
        log.info("Starting Testing")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=None)
        test_metrics = trainer.callback_metrics
    else:
        log.info("Predict not yet implemented")

    metric_dict = {**test_metrics}
    metric_dict = [
        {"name": k, "value": v.item() if hasattr(v, "item") else v}
        for k, v in metric_dict.items()
    ]

    file_path = os.path.join(cfg.paths.output_dir, "finalmetrics.json")
    with open(file_path, "w") as json_file:
        json.dump(metric_dict, json_file)

    utils.close_loggers()


if __name__ == "__main__":
    eval()
