import logging
import torch
from torch import nn
import time


logger = logging.getLogger(__name__)


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear1 = nn.Linear(2, 200)
        self.linear2 = nn.Linear(200, 1)

    def forward(self, batch):
        batch = batch.to(self.linear1.weight.device).float()
        print(batch)
        time.sleep(10)
        res = self.linear1(batch)
        res = self.linear2(res)
        res = res.mean()
        return res

    def save_pretrained(self, outputdir: str):
        print(f"save model to {outputdir}")
