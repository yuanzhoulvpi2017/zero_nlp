from functools import partial
from pathlib import Path
from typing import Any, Dict, List

from datasets import load_dataset
from transformers import LlavaProcessor, AutoProcessor

from .data import build_qaimage, TrainLLavaModelCollator, QaImageOutput
import requests
import torch
import random


def preprocess_sub_data(dataset_dir, examples):
    image_path = examples["image"]

    image_path = str(dataset_dir.joinpath("images_dl").joinpath(image_path))

    conversations = [i for i in examples["conversations"]]
    human_input = conversations[0].get("value")
    chatbot_output = conversations[1].get("value")

    examples["image_path"] = image_path
    examples["human_input"] = human_input
    examples["chatbot_output"] = chatbot_output

    return examples


def preprocess_convert2vector(model_processor, examples):
    result = build_qaimage(
        model_processor,
        examples["human_input"],
        examples["chatbot_output"],
        examples["image_path"],
    )
    examples["q_input_ids"] = result.q_input_ids
    examples["pixel_values"] = result.pixel_values
    examples["a_input_ids"] = result.a_input_ids

    return examples


class SendDatasetByWeb:
    def __init__(
        self, model_name_or_path: str, dataset_dir: str, cache_dir: str, num_proc: int
    ) -> None:

        # 加载默认的processor
        self.model_processor = LlavaProcessor.from_pretrained(model_name_or_path)

        # 获得当前的数据位置
        self.dataset_dir = Path(dataset_dir)

        # 创建数据集
        self.rawdataset_sub_data1 = self.build_dataset(cache_dir, num_proc)

        # 创建随机映射表（在训练阶段，不需要再把数据打乱了）
        self.random_map = self.build_random_idmap()

    def build_random_idmap(self):
        random.seed(42)
        data_size = len(self)
        random_id_list = random.choices(range(data_size), k=data_size)
        random_map = {k: v for k, v in enumerate(random_id_list)}
        return random_map
        


    def build_dataset(self, cache_dir, num_proc):
        rawdataset = load_dataset(
            "json",
            data_files={"train": [str(self.dataset_dir.joinpath("chat.json"))]},
            cache_dir=cache_dir,  # "data/cache_data",
        )

        preprocess_convert2vector_partial = partial(
            preprocess_convert2vector, self.model_processor
        )

        preprocess_sub_data_partial = partial(preprocess_sub_data, self.dataset_dir)

        rawdataset = rawdataset["train"]  # .select(range(10))

        rawdataset_sub_data1 = rawdataset.map(
            function=preprocess_sub_data_partial, num_proc=num_proc, batch_size=3
        )

        rawdataset_sub_data1 = self.rawdataset_sub_data1.map(
            function=preprocess_convert2vector_partial, num_proc=num_proc, batch_size=3
        )
        return rawdataset_sub_data1

    def __len__(self) -> int:
        return len(self.rawdataset_sub_data1)

    def __getitem__(self, index) -> Dict:
        return self.rawdataset_sub_data1[index]


class DatasetReceiveByWeb:
    def __init__(self, host_ip: str = "0.0.0.0"):
        self.host_ip = host_ip

    def __len__(self):
        return self.get_len_from_web(self.host_ip)

    def __getitem__(self, index):
        data = self.get_slice_from_web(index, self.host_ip)
        return data

    def get_slice_from_web(self, index: int, host: str = "0.0.0.0"):  #

        web = requests.get(url=f"http://{host}:7001/slice", params={"index": index})
        json_data = web.json()
        return json_data

    def get_len_from_web(self, host: str = "0.0.0.0"):
        web = requests.get(url=f"http://{host}:7001/len")
        # json_data = web.json()
        return web.json()


class TrainLlavaModelCollatorByWeb(TrainLLavaModelCollator):
    def __call__(self, features: List) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        labels_list = []
        pixel_values = []
        max_input_len_list = []

        for feature in features:
            qaimage_output = QaImageOutput(
                q_input_ids=torch.tensor(feature["q_input_ids"]),
                pixel_values=torch.tensor(feature["pixel_values"]),
                a_input_ids=torch.tensor(feature["a_input_ids"]),
            )

            # build_qaimage(
            #     self.processor, feature[0], feature[1], feature[2]
            # )
            temp_input_ids, temp_labels = self.convert_one_piece(
                qaimage_output.q_input_ids, qaimage_output.a_input_ids
            )
            max_input_len_list.append(temp_input_ids.shape[1])
            input_ids_list.append(temp_input_ids)
            labels_list.append(temp_labels)
            pixel_values.append(qaimage_output.pixel_values)

        max_input_len = max(max_input_len_list)

        final_input_ids = torch.concat(
            [
                torch.concat(
                    [
                        torch.full(
                            (1, max_input_len - max_input_len_list[index]),
                            self.processor.tokenizer.pad_token_id,
                        ),
                        value,
                    ],
                    axis=1,
                )
                for index, value in enumerate(input_ids_list)
            ]
        )
        final_labels = torch.concat(
            [
                torch.concat(
                    [
                        torch.full(
                            (1, max_input_len - max_input_len_list[index]),
                            self.ingnore_index,
                        ),
                        value,
                    ],
                    axis=1,
                )
                for index, value in enumerate(labels_list)
            ]
        )
        final_pixel_values = torch.concat(pixel_values, axis=0)
        attention_mask = torch.ones_like(final_input_ids)
        attention_mask[final_input_ids == self.processor.tokenizer.pad_token_id] = 0
        return {
            "input_ids": final_input_ids,
            "labels": final_labels,
            "pixel_values": final_pixel_values,
            "attention_mask": attention_mask,
        }


__all__ = ["TrainLlavaModelCollatorByWeb", "SendDatasetByWeb", "DatasetReceiveByWeb"]

if __name__ == "__main__":
    from train_llava.data_websend import (
        DatasetReceiveByWeb,
        TrainLlavaModelCollatorByWeb,
    )
    from transformers import LlavaProcessor

    processor = LlavaProcessor.from_pretrained("test_model/model001")

    web_dataset = DatasetReceiveByWeb("10.136.0.65")
    len(web_dataset)

    tlmcw = TrainLlavaModelCollatorByWeb(processor, -100)
    result = tlmcw([web_dataset[0], web_dataset[1]])
    print(result)
