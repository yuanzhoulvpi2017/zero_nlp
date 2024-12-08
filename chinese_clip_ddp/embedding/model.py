import torch
from PIL import Image
from torch import nn
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
from .vit4embedding import ViTForEmbedding
from pathlib import Path
import torch
import torch.distributed as dist
import torch.distributed.nn as dist_nn
from typing import Optional, List


class AllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        world_size = dist.get_world_size()
        ctx.world_size = world_size
        tensors_gather = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(tensors_gather, tensor)
        ctx.rank = dist.get_rank()
        return torch.cat(tensors_gather, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        # 将梯度拆分并获取当前进程对应的部分
        grad_input = grad_output.chunk(ctx.world_size, dim=0)[ctx.rank]
        return grad_input


class TextModelEmbedding(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        device: str,
    ) -> None:
        super(TextModelEmbedding, self).__init__()

        self.model_name_or_path = model_name_or_path

        # Load model from HuggingFace Hub
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.device = device
        self.model.to(self.device)

    def forward(
        self,
        encoded_input: Optional[dict],
        sentences: list[str],
        normalize_embeddings: bool = False,
    ):
        if encoded_input is None:
            # Tokenize sentences
            encoded_input = self.tokenizer(
                sentences,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )

        for i in encoded_input.keys():
            encoded_input[i] = encoded_input[i].to(self.device)

        model_output = self.model(**encoded_input)
        # Perform pooling. In this case, cls pooling.
        sentence_embeddings = model_output[0][:, 0]
        # normalize embeddings
        if normalize_embeddings:
            sentence_embeddings = torch.nn.functional.normalize(
                sentence_embeddings, p=2, dim=1
            )
        return sentence_embeddings

    def save(self, output_dir):
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)


class ImageModelEmbedding(nn.Module):
    def __init__(self, model_name_or_path: str, device: str) -> None:
        super(ImageModelEmbedding, self).__init__()
        self.image_processor = AutoImageProcessor.from_pretrained(model_name_or_path)
        self.model = ViTForEmbedding.from_pretrained(model_name_or_path)
        self.device = device
        self.model.to(self.device)

    def forward(
        self,
        encoded_input: Optional[dict],
        images: List[Image.Image],
        normalize_embeddings: bool = False,
    ):
        if encoded_input is None:
            encoded_input = self.image_processor(images=images, return_tensors="pt")

        for k in encoded_input.keys():
            encoded_input[k] = encoded_input[k].to(self.device)
        last_hidden_states, pool_output = self.model(**encoded_input)
        # last_hidden_states = outputs.last_hidden_state[:, 0]
        pool_output[:, : self.model.config.hidden_size] += last_hidden_states

        if normalize_embeddings:
            pool_output = torch.nn.functional.normalize(pool_output, p=2, dim=1)
        return pool_output

    def save(self, output_dir):
        self.model.save_pretrained(output_dir)
        self.image_processor.save_pretrained(output_dir)


def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(
        logits, torch.arange(len(logits), device=logits.device)
    )


# Copied from transformers.models.clip.modeling_clip.clip_loss
def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


def _freeze_params(module):
    for param in module.parameters():
        param.requires_grad = False


class TextImageEmbeddingModel4LossV1(nn.Module):
    def __init__(
        self,
        text_model_name_or_path: str,
        image_model_name_or_path: str,
        device: str,
        logit_scale_init_value: float = 2.6592,
    ) -> None:
        super(TextImageEmbeddingModel4LossV1, self).__init__()
        self.text_model_embedding = TextModelEmbedding(text_model_name_or_path, device)
        self.image_model_embedding = ImageModelEmbedding(
            image_model_name_or_path, device
        )
        self.logit_scale = nn.Parameter(torch.tensor(logit_scale_init_value))

        # freeze text_model_embedding
        _freeze_params(self.text_model_embedding)

    def calc_loss(
        self, image_embeds: torch.Tensor, text_embeds: torch.Tensor
    ) -> torch.Tensor:
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale

        loss = clip_loss(logits_per_text)
        return loss

    def forward(self, input_ids, attention_mask, pixel_values, **kwargs):
        text_query = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        image_query = {"pixel_values": pixel_values}

        text_embeddings = self.text_model_embedding(
            encoded_input=text_query, sentences=None, normalize_embeddings=False
        )

        image_embeddings = self.image_model_embedding(
            encoded_input=image_query, images=None, normalize_embeddings=False
        )

        loss = self.calc_loss(image_embeddings, text_embeddings)

        return loss

    def save(self, output_dir):
        self.image_model_embedding.save(output_dir)


class TextImageEmbeddingModel4LossV2(TextImageEmbeddingModel4LossV1):
    """
    使用自定义算子进行梯度聚合
    """

    def forward(self, input_ids, attention_mask, pixel_values, **kwargs):
        text_query = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        image_query = {"pixel_values": pixel_values}

        text_embeddings = self.text_model_embedding(
            encoded_input=text_query, sentences=None, normalize_embeddings=False
        )

        image_embeddings = self.image_model_embedding(
            encoded_input=image_query, images=None, normalize_embeddings=False
        )

        all_text_embeddings = AllGather.apply(text_embeddings.contiguous())
        all_image_embeddings = AllGather.apply(image_embeddings.contiguous())

        loss = self.calc_loss(all_text_embeddings, all_image_embeddings)

        return loss


class TextImageEmbeddingModel4LossV3(TextImageEmbeddingModel4LossV1):
    """
    使用dist_nn.all_gather进行梯度聚合
    """

    def forward(self, input_ids, attention_mask, pixel_values, **kwargs):
        text_query = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        image_query = {"pixel_values": pixel_values}

        text_embeddings = self.text_model_embedding(
            encoded_input=text_query, sentences=None, normalize_embeddings=False
        )

        image_embeddings = self.image_model_embedding(
            encoded_input=image_query, images=None, normalize_embeddings=False
        )

        all_text_embeddings = dist_nn.all_gather(text_embeddings)
        all_text_embeddings = torch.cat(all_text_embeddings, dim=0)

        all_image_embeddings = dist_nn.all_gather(image_embeddings)
        all_image_embeddings = torch.cat(all_image_embeddings, dim=0)

        loss = self.calc_loss(all_text_embeddings, all_image_embeddings)

        return loss


class TextImageEmbeddingModel4LossV4(TextImageEmbeddingModel4LossV1):
    """
    使用常规思路进行梯度聚合
    """

    def forward(self, input_ids, attention_mask, pixel_values, **kwargs):
        text_query = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        image_query = {"pixel_values": pixel_values}

        text_embeddings = self.text_model_embedding(
            encoded_input=text_query, sentences=None, normalize_embeddings=False
        ).contiguous()

        image_embeddings = self.image_model_embedding(
            encoded_input=image_query, images=None, normalize_embeddings=False
        ).contiguous()

        # 获取当前进程的rank和总进程数
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        text_embeddings_list = [
            torch.zeros_like(text_embeddings) for _ in range(world_size)
        ]
        image_embeddings_list = [
            torch.zeros_like(image_embeddings) for _ in range(world_size)
        ]

        # 使用all_gather收集所有进程的text_embeddings和image_embeddings
        dist.all_gather(text_embeddings_list, text_embeddings)
        dist.all_gather(image_embeddings_list, image_embeddings)

        # 将收集到的嵌入拼接在一起
        all_text_embeddings = torch.cat(text_embeddings_list, dim=0)
        all_image_embeddings = torch.cat(image_embeddings_list, dim=0)

        loss = self.calc_loss(all_text_embeddings, all_image_embeddings)

        return loss


MODEL_TRAIN_MAP = {
    "base": TextImageEmbeddingModel4LossV1,
    "custom": TextImageEmbeddingModel4LossV2,
    "gather_nn": TextImageEmbeddingModel4LossV3,
    # 这个其实很诡异：在有的版本上，这个跑直接报错，提示梯度传播有问题；但是在有的版本上，可以跑，但是loss不下降，非常恐怖～
    "gather": TextImageEmbeddingModel4LossV4,
}
