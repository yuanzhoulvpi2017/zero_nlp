import logging
from transformers import Trainer
from typing import Optional
from .model import MyModel
from transformers import is_datasets_available
import datasets

from torch.utils.data import DataLoader
import torch
from transformers.trainer_utils import seed_worker

from torch.utils.data import Sampler,SequentialSampler

logger = logging.getLogger(__name__)


class MyTrainer(Trainer):
    def _save(self, output_dir: Optional[str] = None):
        self.model.save_pretrained(output_dir)

    def compute_loss(self, model: MyModel, inputs):
        return model(**inputs)
    
    def _get_train_sampler(self):
        return SequentialSampler(self.train_dataset)


    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(
                train_dataset, description="training"
            )
        else:
            data_collator = self._get_collator_with_removed_columns(
                data_collator, description="training"
            )

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
