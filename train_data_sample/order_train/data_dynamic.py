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
import numpy as np

def get_all_datapath(dir_name: str) -> List[str]:
    """
    接收一个目录名称（dir_name）作为输入，并返回该目录及其子目录中所有文件路径的列表
    """
    if isinstance(dir_name, Path):
        dir_name = str(dir_name)

    all_file_list = []
    for root, dir, file_name in os.walk(dir_name):
        for temp_file in file_name:
            standard_path = f"{root}/{temp_file}"
            all_file_list.append(standard_path)
    return all_file_list

class TrainDatasetForOrder(Dataset):
    """
    从给定目录加载两种类型的数据，specdata和normdata
    代表垂类的专业数据集和通用的数据集
    """
    def __init__(self, dataset_dir: str, cache_dir: str = "cache_data", start_ratio: float = 0.5) -> None:
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.cache_dir = cache_dir

        # 两种数据集
        self.spec_data = self.load_a_type_data(type_name="specdata")
        self.norm_data = self.load_a_type_data(type_name="normdata")

        # 计算数据总长
        self.len_spec = len(self.spec_data)
        self.len_norm = len(self.norm_data)
        self.len_total = self.len_norm + self.len_spec

        # 初始比例设置为0.5，表示norm_data为spec_data的比例
        self.norm_ratio = start_ratio  

    def set_sampling_ratio(self, ratio):
        """动态调整norm_data和spec_data的抽样比例"""
        self.norm_ratio = ratio

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
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=cache_dir,
        )["train"]
        return raw_datasets

    def __len__(self):
        return self.len_total

    def __getitem__(self, index):
        # 生成随机数决定采样的数据集类型
        if np.random.rand() < (self.norm_ratio / (1 + self.norm_ratio)):
            return self.get_norm_item(index)
        else:
            return self.get_spec_item(index)
    
    def get_norm_item(self, index):
        """从norm_data中获取数据项的逻辑"""
        norm_index = index % self.len_norm  # 循环使用norm_data
        tempdata = self.norm_data[norm_index]
        tempdata = torch.tensor([200, int(tempdata["text"])])
        return tempdata
    
    def get_spec_item(self, index):
        """从spec_data中获取数据项的逻辑"""
        spec_index = index % self.len_spec  # 循环使用spec_data
        tempdata = self.spec_data[spec_index]
        tempdata = torch.tensor([100, int(tempdata["text"])])
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
