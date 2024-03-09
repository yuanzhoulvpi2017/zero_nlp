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

class DynamicSamplingCallback(TrainerCallback):
    def __init__(self, train_dataset, initial_ratio=0.5, final_ratio=4.5, total_steps=100):
        self.train_dataset = train_dataset
        self.initial_ratio = initial_ratio
        self.final_ratio = final_ratio
        self.total_steps = total_steps
        self.step_count = 0

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """
        # 随着训练进度的增加，抽样比例会从初始比例initial_ratio线性增加到final_ratio
        # 避免灾难性遗忘
        """
        self.step_count += 1
        new_ratio = self.initial_ratio + (self.final_ratio - self.initial_ratio) * (self.step_count / self.total_steps)
        self.train_dataset.set_sampling_ratio(new_ratio)

class MyTrainer(Trainer):
    def train(self):
        # 在开始训练前增加了DynamicSamplingCallback回调
        dynamic_callback = DynamicSamplingCallback(
            self.train_dataset,
            initial_ratio=0.5,
            final_ratio=4.5,
            total_steps=len(self.get_train_dataloader()) * self.args.num_train_epochs
        )
        self.add_callback(dynamic_callback)
        super().train()
    def _save(self, output_dir: Optional[str] = None):
        self.model.save_pretrained(output_dir)

    def compute_loss(self, model: MyModel, inputs):
        return model(**inputs)
    
    def _get_train_sampler(self):
        return SequentialSampler(self.train_dataset)


    def get_train_dataloader(self) -> DataLoader:
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
