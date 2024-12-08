from PIL import Image
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Any
import torch
from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize
from torchvision.transforms.functional import InterpolationMode
from torchvision.io import ImageReadMode, read_image


@dataclass
class ImageTextData:
    image_path: Path
    text: str
    other: Any = field(default=None)


class ImageTextDataset(Dataset):
    def __init__(self, data_size: str, language: str) -> None:
        super().__init__()
        self.data_size = data_size
        self.language = language
        if data_size == "small":
            self.image_dir = Path("data/jackyhate/text-to-image-2M/data_1024_10K")
            self.text_file = Path("data/jackyhate/text-to-image-2M/data_1024_10K.json")

        elif data_size == "large":
            self.image_dir = Path("data/jackyhate/unzip2mdata")
            self.text_file = Path(
                "data/jackyhate/text-to-image-2M/big_tran_zh_data.json"
            )

        self.text_data = self.load_text_data()

    def load_text_data(self):
        with open(self.text_file, "r") as fin:
            all_data = fin.readlines()
            all_data = [
                json.loads(data) for data in tqdm(all_data, desc="loading text data")
            ]
        return all_data

    def __len__(self) -> int:
        return len(self.text_data)

    def __getitem__(self, idx) -> ImageTextData:
        temp_data = self.text_data[idx]
        image_path = self.image_dir / f"{temp_data.get('file_path')}.jpg"

        if self.language == "zh":
            text = temp_data.get("zh_prompt", None)
            if text is None:
                text = temp_data.get("prompt", None)

        else:
            text = temp_data.get("prompt", None)

        return ImageTextData(image_path, text, other=temp_data)


class Transform(torch.nn.Module):
    def __init__(self, image_size, mean, std):
        super().__init__()
        self.transforms = torch.nn.Sequential(
            Resize(
                [image_size], interpolation=InterpolationMode.BICUBIC, antialias=True
            ),
            CenterCrop(image_size),
            ConvertImageDtype(torch.float),
            Normalize(mean, std),
        )

    def forward(self, x) -> torch.Tensor:
        """`x` should be an instance of `PIL.Image.Image`"""
        with torch.no_grad():
            x = self.transforms(x)
        return x


class ImageTextDataCollator:
    def __init__(self, image_size, mean, std, tokenizer, max_seq_length):
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.image_transformations = Transform(
            self.image_size,
            self.mean,
            self.std,
        )
        self.image_transformations = torch.jit.script(self.image_transformations)

    def __call__(self, batch: List[ImageTextData]) -> dict[str, Any]:
        # 需要过滤一些损坏的图片
        batch = [
            example
            for example in batch
            if self.filter_corrupt_images(example.image_path)
        ]
        pixel_values = torch.stack([
            self.transform_image(example.image_path) for example in batch
        ])
        text_outputs = self.transform_text([example.text for example in batch])

        return {"pixel_values": pixel_values, **text_outputs}

    def transform_image(self, image_path: Path) -> torch.Tensor:
        images = read_image(str(image_path), mode=ImageReadMode.RGB)
        pixel_values = self.image_transformations(images)
        return pixel_values

    def transform_text(self, text: List[str]) -> dict[str, Any]:
        text_inputs = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_seq_length,
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
        }

    def filter_corrupt_images(self, image_path: Path):
        """remove problematic images"""
        try:
            read_image(str(image_path), mode=ImageReadMode.RGB)
            return True
        except Exception:
            return False
