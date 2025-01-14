from datasets import Audio

from birdset.datamodule.components.transforms import BirdSetTransformsWrapper
from birdset.datamodule.base_datamodule import (
    BaseDataModuleHF,
    DatasetConfig,
    LoadersConfig,
)

from audioprotopnet.helpers import get_label_to_category_mapping_from_hf_dataset


class ESC50DataModule(BaseDataModuleHF):
    def __init__(
        self,
        dataset: DatasetConfig = DatasetConfig(),
        loaders: LoadersConfig = LoadersConfig(),
        transforms: BirdSetTransformsWrapper = BirdSetTransformsWrapper(),
        mapper: None = None,
    ):
        super().__init__(
            dataset=dataset, loaders=loaders, transforms=transforms, mapper=mapper
        )

        self.label_to_category_mapping = None

    def _preprocess_data(self, dataset):
        dataset = dataset.cast_column(
            column="audio",
            feature=Audio(
                sampling_rate=self.dataset_config.sampling_rate,
                mono=True,
                decode=True,
            ),
        )

        self.label_to_category_mapping = get_label_to_category_mapping_from_hf_dataset(
            dataset=dataset["train"]
        )

        if self.event_mapper is not None:
            dataset = dataset["train"].map(
                self.event_mapper,
                batched=True,
                batch_size=300,
                load_from_cache_file=True,
                num_proc=self.dataset_config.n_workers,
            )
        dataset = dataset.rename_column("target", "labels")
        dataset = dataset.select_columns(["audio", "labels"])
        return dataset
