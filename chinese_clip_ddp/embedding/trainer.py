import os
from typing import Dict, List, Optional, Sequence

import torch
from transformers import Trainer




class HzTrainer(Trainer):
    # model: TextEmbeddingModel4Loss2
    def compute_loss(
        self,
        model,
        inputs,
        **kwargs,
    ):
        loss = model(**inputs)
        return loss

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.model.save(output_dir)
