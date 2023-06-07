from tqdm import tqdm
import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
from datasets import load_dataset
from typing import List
import os
import logging
from transformers import DataCollatorForSeq2Seq, default_data_collator, DataCollatorForLanguageModeling
from functools import partial
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'
logger = logging.getLogger(__name__)


def get_all_datapath(dir_name: str) -> List[str]:
    all_file_list = []
    # all_file_size = []

    for (root, dir, file_name) in os.walk(dir_name):
        for temp_file in file_name:
            standard_path = f"{root}/{temp_file}"

            all_file_list.append(standard_path)

    return all_file_list


def load_dataset_from_path(data_path: Optional[str] = None,
                           cache_dir: Optional[str] = "cache_data",
                           data_file_number:Optional[int] = 2) -> Dataset:
    all_file_list = get_all_datapath(data_path)[:data_file_number]
    data_files = {'train': all_file_list}
    extension = all_file_list[0].split(".")[-1]

    logger.info("load files %d number", len(all_file_list))

    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=cache_dir,
    )['train']
    return raw_datasets


IGNORE_INDEX = -100


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={
        "help": "Path to the training data."})
    data_num_limit:int = field(default=None, metadata={
        "help":"the numbers of data file"
    })
    data_proc_num:int = field(default=None, metadata={
        "help":"the numbers of process"
    })



@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def make_train_dataset(tokenizer: transformers.PreTrainedTokenizer, data_path: str, data_file_number:int, data_proc_num:int) -> Dataset:
    logging.warning("Loading data...")

    dataset = load_dataset_from_path(
        data_path=data_path,
        data_file_number=data_file_number
    )
    logging.warning("Formatting inputs...")

    def generate_sources_targets(examples: Dict, tokenizer: transformers.PreTrainedTokenizer):
        ins_data = examples['content']

        input_output = tokenizer(ins_data,
                                 return_tensors="pt",
                                 padding="longest",
                                 max_length=tokenizer.model_max_length-1,
                                 truncation=True)
        examples['input_ids'] = input_output['input_ids']
        return examples

    generate_sources_targets_p = partial(
        generate_sources_targets, tokenizer=tokenizer)

    dataset = dataset.map(
        function=generate_sources_targets_p,
        batched=True,
        desc="Running tokenizer on train dataset",
        num_proc=data_proc_num
    ).shuffle()
    return dataset


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        # device_map='auto',
        torch_dtype=torch.bfloat16

    )
    # setattr(model, "is_parallelizable", True)
    # setattr(model, "model_parallel", True)

    # model.is_parallelizable = True
    # model.model_parallel = True
    torch.cuda.empty_cache()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )

    train_dataset = make_train_dataset(
        tokenizer=tokenizer, data_path=data_args.data_path, data_file_number=data_args.data_num_limit, data_proc_num=data_args.data_proc_num)
    train_dataset = train_dataset.remove_columns(
        ['uniqueKey', 'title', 'titleUkey', 'dataType', 'id', 'content'])

    data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8
        )

    trainer = Trainer(model=model,
                      tokenizer=tokenizer,
                      args=training_args,
                      train_dataset=train_dataset,
                      eval_dataset=None,
                      data_collator=data_collator)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )
    train()
