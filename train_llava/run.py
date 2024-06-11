import copy
import logging
import logging
import os
import torch
import transformers
from dataclasses import dataclass, field
from datasets import load_dataset
from functools import partial
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import DataCollatorForSeq2Seq, Trainer
from typing import Dict, Optional, Sequence, List
from train_llava.data import LlavaDataset, TrainLLavaModelCollator
from transformers import TrainingArguments, Trainer
from transformers import AutoProcessor, LlavaForConditionalGeneration, LlavaProcessor

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="test_model/model001")
    use_lora: Optional[bool] = field(default=False)


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    # source_length: int = field(default=128)
    # target_length: int = field(default=512)


def load_model_processor(modelargs: ModelArguments):
    model = LlavaForConditionalGeneration.from_pretrained(
        modelargs.model_name_or_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    processor = LlavaProcessor.from_pretrained(modelargs.model_name_or_path)
    
    if modelargs.use_lora:
        logging.warning("Loading model to Lora")

        from peft import LoraConfig, get_peft_model

        LORA_R = 32
        # LORA_ALPHA = 16
        LORA_DROPOUT = 0.05
        TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

        config = LoraConfig(
            r=LORA_R,
            # lora_alpha=LORA_ALPHA,
            target_modules=TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
        )
        # model = model.to(torch.bfloat16)
        model = get_peft_model(model, config)
        # peft_module_casting_to_bf16(model)
        model.print_trainable_parameters()


    return model, processor


def load_dataset(dataargs: DataArguments):
    llava_dataset = LlavaDataset(
        dataargs.data_path  # "data/liuhaotian/LLaVA-CC3M-Pretrain-595K"
    )
    return llava_dataset



def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model, processor = load_model_processor(model_args)
    data_collator = TrainLLavaModelCollator(processor, -100)
    train_dataset = load_dataset(data_args)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    train()
