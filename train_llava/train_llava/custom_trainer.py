from transformers import Trainer
from transformers.trainer_pt_utils import ShardSampler
from typing import Optional
import torch
from transformers.trainer_utils import has_length


class WebTrainer(Trainer):

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        return ShardSampler(self.train_dataset)
