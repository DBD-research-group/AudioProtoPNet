import math

import datasets
import hydra
import lightning
import torch
import wandb

from birdset.modules.losses import load_loss
from birdset.modules.metrics import load_metrics


class BaseModule(lightning.LightningModule):
    def __init__(
        self,
        network,
        output_activation,
        loss,
        optimizer,
        lr_scheduler,
        metrics,
        logging_params,
        num_epochs,
        len_trainset,
        batch_size,
        task,
        class_weights_loss,
        label_counts,
        num_gpus,
        prediction_table=None,
    ):
        super().__init__()

        # partial
        self.num_epochs = num_epochs
        self.len_trainset = len_trainset
        self.batch_size = batch_size
        self.task = task

        self.model = network["model"]
        self.opt_params = optimizer
        self.lrs_params = lr_scheduler
        self.num_gpus = num_gpus

        self.loss = load_loss(loss, class_weights_loss, label_counts)
        self.output_activation = hydra.utils.instantiate(
            output_activation, _partial_=True
        )
        self.logging_params = logging_params

        self.metrics = load_metrics(metrics)
        self.train_metric = self.metrics["main_metric"].clone()
        self.train_add_metrics = self.metrics["add_metrics"].clone(prefix="train/")

        self.valid_metric = self.metrics["main_metric"].clone()
        self.valid_metric_best = self.metrics["val_metric_best"].clone()
        self.valid_add_metrics = self.metrics["add_metrics"].clone(prefix="val/")

        self.test_metric = self.metrics["main_metric"].clone()
        self.test_add_metrics = self.metrics["add_metrics"].clone(prefix="test/")
        self.test_complete_metrics = self.metrics["eval_complete"].clone(prefix="test/")

        self.torch_compile = network["torch_compile"]
        self.model_name = network["model_name"]

        self.save_hyperparameters()

        self.test_targets = []
        self.test_preds = []

        self.prediction_table = prediction_table

        if self.model.pretrain_info and self.model.pretrain_info.get(
            "hf_pretrain_name"
        ):
            print("Masking Logits")
            self.pretrain_dataset = self.model.pretrain_info["hf_pretrain_name"]
            self.hf_path = self.model.pretrain_info["hf_path"]
            self.hf_name = self.model.pretrain_info["hf_name"]
            pretrain_classlabels = datasets.load_dataset_builder(
                self.hf_path, self.pretrain_dataset
            ).info.features["ebird_code"]
            dataset_classlabels = datasets.load_dataset_builder(
                self.hf_path, self.hf_name
            ).info.features["ebird_code"]
            self.class_mask = [
                pretrain_classlabels.names.index(i) for i in dataset_classlabels.names
            ]

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.opt_params, params=self.parameters(), _convert_="partial"
        )

        if self.lrs_params.get("scheduler"):
            num_training_steps = math.ceil(
                (self.num_epochs * self.len_trainset) / self.batch_size * self.num_gpus
            )
            # TODO: Handle the case when drop_last=True more explicitly

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

    def model_step(self, batch, batch_idx):
        logits = self.forward(**batch)
        if self.class_mask and (
            not self.model.pretrain_info.valid_test_only or not self.trainer.training
        ):
            logits = logits[:, self.class_mask]
        loss = self.loss(logits, batch["labels"])
        preds = self.output_activation(logits)
        return loss, preds, batch["labels"]

    def on_train_start(self):
        self.valid_metric_best.reset()

    def training_step(self, batch, batch_idx):
        train_loss, preds, targets = self.model_step(batch, batch_idx)
        self.log(
            f"train/{self.loss.__class__.__name__}",
            train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        # Too slow!
        # self.train_metric(preds, targets.int())
        # self.log(
        #     f"train/{self.train_metric.__class__.__name__}",
        #     self.train_metric,
        #     **self.logging_params,
        # )

        # self.train_add_metrics(preds, targets)
        # self.log_dict(self.train_add_metrics, **self.logging_params)

        return {"loss": train_loss}

    def validation_step(self, batch, batch_idx):
        val_loss, preds, targets = self.model_step(batch, batch_idx)

        self.log(
            f"val/{self.loss.__class__.__name__}",
            val_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        # self.valid_metric(preds, targets.int())
        # self.log(
        #     f"val/{self.valid_metric.__class__.__name__}",
        #     self.valid_metric,
        #     **self.logging_params,
        # )

        # Too slow!
        # self.valid_add_metrics(preds, targets.int())
        # self.log_dict(self.valid_add_metrics, **self.logging_params)
        return {"loss": val_loss, "preds": preds, "targets": targets}

    # def on_validation_epoch_end(self):
    #     valid_metric = self.valid_metric.compute()  # get current valid metric
    #     self.valid_metric_best(valid_metric)  # update best so far valid metric
    #
    #     self.log(
    #         f"val/{self.valid_metric.__class__.__name__}_best",
    #         self.valid_metric_best.compute(),
    #     )

    def test_step(self, batch, batch_idx):
        test_loss, preds, targets = self.model_step(batch, batch_idx)

        # save targets and predictions for test_epoch_end
        self.test_targets.append(targets.detach().cpu())
        self.test_preds.append(preds.detach().cpu())

        self.log(
            f"test{self.loss.__class__.__name__}",
            test_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.test_metric(preds, targets.int())
        self.log(
            f"test/{self.test_metric.__class__.__name__}",
            self.test_metric,
            **self.logging_params,
        )

        self.test_add_metrics(preds, targets.int())
        self.log_dict(self.test_add_metrics, **self.logging_params)

        return {"loss": test_loss, "preds": preds, "targets": targets}

    def setup(self, stage):
        if self.torch_compile and stage == "fit":
            print("COMPILE")
            self.model = torch.compile(self.model)

    def on_test_epoch_end(self):
        if self.task == "multilabel":
            test_targets = torch.cat(self.test_targets).int()
            test_preds = torch.cat(self.test_preds)
            # self.test_complete_metrics(test_preds, test_targets)

            log_dict = {}

            # Rename cmap to cmap5!
            for metric_name, metric in self.test_complete_metrics.named_children():
                value = metric(test_preds, test_targets)
                log_dict[f"test/{metric_name}"] = value

            self.log_dict(log_dict, **self.logging_params)

            if self.prediction_table:
                self._wandb_prediction_table(test_preds, test_targets)
        else:
            pass

    def _wandb_prediction_table(self, preds, targets):
        top5_values_preds, top5_indices_preds = preds.topk(dim=1, k=5, sorted=True)

        top5_values_preds = top5_values_preds.cpu().numpy()
        top5_indices_preds = top5_indices_preds.cpu().numpy()

        indices_targets = (targets == 1).nonzero(as_tuple=False)
        multilabel_list = [[] for _ in range(targets.size(0))]
        for idx in indices_targets:
            row, col = idx.tolist()
            multilabel_list[row].append(col)

        # Convert the indices to strings
        top5_indices_preds_str = [
            ", ".join(map(str, indices)) for indices in top5_indices_preds
        ]
        multilabel_list_str = [
            ", ".join(map(str, indices)) for indices in multilabel_list
        ]

        # Prepare the data for the wandb table
        columns = ["Predictions", "Targets"]
        data = [
            [pred_str, tgt_str]
            for pred_str, tgt_str in zip(top5_indices_preds_str, multilabel_list_str)
        ]
        table = wandb.Table(data=data, columns=columns)

        # Log the table to wandb
        wandb.log({"Top 5 Predictions vs Targets": table})
