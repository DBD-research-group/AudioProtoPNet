import os

from birdset import utils
import hydra
import lightning as L
from omegaconf import OmegaConf
import pyrootutils

from audioprotopnet.preprocessing.normalization.standardization import (
    calculate_mean_std_from_dataloader,
)


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
    "config_name": "main_benchmarks.yaml",
}


# @utils.register_custom_resolvers(**_HYDRA_PARAMS)
@hydra.main(**_HYDRA_PARAMS)
def calculate_normalization_parameters(cfg):
    log.info("Using config: \n%s", OmegaConf.to_yaml(cfg))

    log.info(f"Dataset path: <{os.path.abspath(cfg.paths.dataset_path)}>")
    os.makedirs(cfg.paths.dataset_path, exist_ok=True)

    log.info(f"Log path: <{os.path.abspath(cfg.paths.log_dir)}>")
    os.makedirs(cfg.paths.log_dir, exist_ok=True)

    log.info(f"Root Dir:<{os.path.abspath(cfg.paths.log_dir)}>")
    log.info(f"Work Dir:<{os.path.abspath(cfg.paths.work_dir)}>")
    log.info(f"Output Dir:<{os.path.abspath(cfg.paths.output_dir)}>")

    log.info(f"Seed everything with <{cfg.seed}>")
    L.seed_everything(cfg.seed)

    # Setup data
    log.info(f"Instantiate datamodule <{cfg.datamodule._target_}>")
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    # Set the normalization in the config file to False, since we do not want normalized data for the mean and
    # standard deviation calculations.
    datamodule.transforms.preprocessing.normalize_waveform = False
    datamodule.transforms.preprocessing.normalize_spectrogram = False

    # Set the augmentations in the config file to None, since we do not want augmentations for the mean and
    # standard deviation calculations.
    datamodule.transforms.waveform_augmentations = []
    datamodule.transforms.spectrogram_augmentations = []

    datamodule.prepare_data()

    datamodule.setup(stage="fit")

    mean, std = calculate_mean_std_from_dataloader(
        dataloader=datamodule.train_dataloader()
    )

    log.info(f"Mean: {mean} | Standard deviation: {std}")


if __name__ == "__main__":
    calculate_normalization_parameters()
