import math
import os
import random
from dataclasses import dataclass
from typing import Any, List, Tuple, Dict

import datasets
import torch
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding
from transformers import PreTrainedTokenizer, BatchEncoding
from pathlib import Path
from typing import Optional
import os
from datasets import load_dataset


def get_all_datapath(dir_name: str) -> List[str]:
    if isinstance(dir_name, Path):
        dir_name = str(dir_name)
    all_file_list = []

    for root, dir, file_name in os.walk(dir_name):
        for temp_file in file_name:
            standard_path = f"{root}/{temp_file}"

            all_file_list.append(standard_path)

    return all_file_list


class TrainDatasetForOrder(Dataset):
    def __init__(self, dataset_dir: str, cache_dir: str = "cache_data") -> None:
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.cache_dir = cache_dir

        self.spec_data = self.load_a_type_data(type_name="specdata")
        self.norm_data = self.load_a_type_data(type_name="normdata")

        self.len_spec = len(self.spec_data)
        self.len_norm = len(self.norm_data)
        self.len_total = self.len_norm + self.len_spec

    def load_a_type_data(self, type_name: str) -> Dataset:
        data1 = self.load_dataset_from_path(
            str(self.dataset_dir.joinpath(type_name)), self.cache_dir
        )
        return data1

    def load_dataset_from_path(
        self, data_path: Optional[str] = None, cache_dir: Optional[str] = "cache_data"
    ) -> Dataset:
        all_file_list = get_all_datapath(data_path)
        data_files = {"train": all_file_list}
        extension = all_file_list[0].split(".")[-1]

        # logger.info("load files %d number", len(all_file_list))

        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=cache_dir,
        )["train"]
        return raw_datasets

    def __len__(self):
        return self.len_total

    def __getitem__(self, index):
        # 这里默认的是通用领域的数量比spec数据更多
        if index < self.len_spec:
            tempdata = self.spec_data[index]
            tempdata = torch.tensor([100, int(tempdata["text"])])
            return tempdata
        else:
            tempdata = self.norm_data[index - self.len_spec]
            tempdata = torch.tensor([200, int(tempdata["text"])])
            return tempdata


@dataclass
class GroupCollator:
    def __call__(self, features: List):
        temp_data = torch.concatenate([i.reshape(1, -1) for i in features], dim=0)
        temp_data = {"batch":temp_data}
        return temp_data


if __name__ == "__main__":
    dataset_test = TrainDatasetForOrder(
        dataset_dir="/data2/hz_data2/train_diaodu_1229/data/test001data"
    )



    groupcollator = GroupCollator()

    res = groupcollator([dataset_test[i] for i in [1, 1300, 10000]], )

    print(res.shape)
