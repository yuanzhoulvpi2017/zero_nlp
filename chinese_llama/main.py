
import logging
import os
import sys
import json
import click
import numpy as np
from datasets import load_dataset
# import jieba
# from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch
from functools import partial

from typing import List, Dict, Optional, Union
import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
    TrainingArguments
)
from trainer import Trainer
# from arguments import ModelArguments, DataTrainingArguments

logger = logging.getLogger(__name__)


def load_tokenizer(model_name_or_path: str = "fastchat/tokenizer"):
    logger.info(f"init tokenizer")
    from fastchat.tokenizer.tokenization_llama_zh import LlamazhTokenizer
    tokenizer = LlamazhTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True)
    return tokenizer


def load_model(tokenizer, model_name_or_path: Optional[str] = None, device_map: Optional[Dict[int, List[int]]] = None):

    from transformers.models.llama import LlamaConfig
    logger.info("init model")

    config = LlamaConfig(vocab_size=tokenizer.__len__(),
                         hidden_size=2048,
                         intermediate_size=5504,  # 11008,
                         num_hidden_layers=32,  # 32,
                         #  num_attention_heads=32,
                         bos_token_id=tokenizer.bos_token_id,
                         eos_token_id=tokenizer.eos_token_id,
                         pad_token_id=tokenizer.pad_token_id,
                         )

    from fastchat.models.llama.modeling_llama_zh import LlamaForCausalLM
    model = LlamaForCausalLM(config=config).to(torch.bfloat16).cuda()
    model.parallelize(device_map=device_map)

    return model


def get_all_datapath(dir_name: str) -> List[str]:
    all_file_list = []
    # all_file_size = []

    for (root, dir, file_name) in os.walk(dir_name):
        for temp_file in file_name:
            standard_path = f"{root}/{temp_file}"

            all_file_list.append(standard_path)

    return all_file_list


def load_dataset_from_path(data_path: Optional[str] = None,
                           cache_dir: Optional[str] = None):

    all_file_list = get_all_datapath(data_path)
    data_files = {'train': all_file_list}
    extension = all_file_list[0].split(".")[-1]

    logger.info("load files %d number", len(all_file_list))

    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=cache_dir,
    )
    return raw_datasets


def load_tokenizer_and_model():

    device_map = {
        0: [0, 1, 2, 3, 4, 5, 6, ],
        1: [7, 8, 9, 10, 11, 12, 13, ],
        2: [14, 15, 16, 17, 18, 19, 20, 21, ],
        3: [22, 23, 24, 25, 26, 27, 28, 29],
        4: [30, 31]
    }
    device_map = {
        0: [0, 1, 2, 3, 4, 5, 6, 7, 8],
        1: [9, 10, 11]
    }
    device_map = {
        0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, ],
        1: [24, 25, 26, 27, 28, 29, 30, 31],
    }

    tokenizer = load_tokenizer()
    model = load_model(tokenizer=tokenizer, device_map=device_map)

    return tokenizer, model


def preprocess_function_(examples: Dict,
                         tokenizer: AutoTokenizer,
                         max_source_length: int = 1024,
                         max_target_length: int = 1024,
                         prompt_column: Optional[str] = 'q',
                         response_column: Optional[str] = 'a',
                         history_column: Optional[str] = None,
                         ignore_pad_token_for_loss=-100,
                         ):
    max_seq_length = max_source_length + max_target_length

    model_inputs = {
        "input_ids": [],
        "labels": [],
    }
    for i in range(len(examples[prompt_column])):
        if examples[prompt_column][i] and examples[response_column][i]:
            query, answer = examples[prompt_column][i], examples[response_column][i]

            if history_column is None:
                prompt = query
            else:
                prompt = ""
                history = examples[history_column][i]
                for turn_idx, (old_query, response) in enumerate(history):
                    prompt += "[Round {}]\n问：{}\n答：{}\n".format(
                        turn_idx, old_query, response)
                prompt += "[Round {}]\n问：{}\n答：".format(
                    len(history), query)

            prompt = prompt


            a_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
            b_ids = tokenizer.encode(text=answer, add_special_tokens=False)

            if len(a_ids) > max_source_length - 1:
                a_ids = a_ids[: max_source_length - 1]

            if len(b_ids) > max_target_length - 2:
                b_ids = b_ids[: max_target_length - 2]

            input_ids = tokenizer.build_inputs_with_special_tokens(
                a_ids, b_ids)

            context_length = input_ids.index(tokenizer.bos_token_id)
            mask_position = context_length - 1
            labels = [-100] * context_length + input_ids[mask_position+1:]

            pad_len = max_seq_length - len(input_ids)
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
            labels = labels + [tokenizer.pad_token_id] * pad_len
            if ignore_pad_token_for_loss:
                labels = [(l if l != tokenizer.pad_token_id else -100)
                          for l in labels]

            model_inputs["input_ids"].append(input_ids)
            model_inputs["labels"].append(labels)

    return model_inputs


def train(*,
          dataset_path: str,
          epochs: int,
          per_device_train_batch_size: int,
          per_device_eval_batch_size: int,
          lr: float,
          seed: int,
          logging_steps: int,
          save_steps: int,
          eval_steps: int,
          test_size: Union[float, int],
          save_total_limit: int,
          local_output_dir: str,
          warmup_steps: int,
          max_source_length: int,
          max_target_length: int,
          gradient_accumulation_steps: int):
    set_seed(seed=seed)
    tokenizer, model = load_tokenizer_and_model()

    dataset = load_dataset_from_path(
        data_path=dataset_path, cache_dir="cache_data")['train']
    preprocess_function = partial(preprocess_function_, tokenizer=tokenizer,
                                  max_source_length=max_source_length,
                                  max_target_length=max_target_length,
                                  prompt_column='q',
                                  response_column='a',
                                  history_column=None,
                                  ignore_pad_token_for_loss=-100
                                  )
    dataset = dataset.map(
        function=preprocess_function,
        batched=True,
        desc="Running tokenizer on train dataset",
        remove_columns=['q', 'a'],
        num_proc=10

    )
    logger.info("Processed dataset has %d rows", dataset.num_rows)
    dataset = dataset.shuffle(seed=seed)
    split_dataset = dataset.train_test_split(test_size=test_size, seed=seed)

    logger.info("Train data size: %d", split_dataset["train"].num_rows)
    logger.info("Test data size: %d", split_dataset["test"].num_rows)

    def print_dataset_example(example):
        print("input_ids", example["input_ids"])
        print("inputs", tokenizer.decode(example["input_ids"]))
        print("label_ids", example["labels"])
        print("labels", tokenizer.decode(example["labels"]))

    print_dataset_example(split_dataset['train'][0])

    label_pad_token_id = - 100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=None,
        padding=False
    )
    training_args = TrainingArguments(
        output_dir=local_output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        fp16=False,
        bf16=True,
        learning_rate=lr,
        num_train_epochs=epochs,
        logging_strategy="steps",
        logging_steps=logging_steps,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        load_best_model_at_end=False,
        report_to="tensorboard",
        disable_tqdm=False,
        remove_unused_columns=False,
        # local_rank=local_rank,
        warmup_steps=warmup_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    logger.info("Instantiating Trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split_dataset['train'],
        eval_dataset=split_dataset['test'],
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    logger.info("Training")
    trainer.train()

    logger.info(f"Saving Model to {local_output_dir}")
    trainer.save_model(output_dir=local_output_dir)


@click.command()
@click.option("--dataset-path", type=str, default="data/opendata")
@click.option("--epochs", type=int, default=3)
@click.option("--per-device-train-batch-size", type=int, default=3)
@click.option("--per-device-eval-batch-size", type=int, default=1)
@click.option("--lr", type=float, default=1e-5)
@click.option("--seed", type=int, default=42)
@click.option("--logging-steps", type=int, default=10)
@click.option("--save-steps", type=int, default=1000)
@click.option("--eval-steps", type=int, default=500)
@click.option("--test-size", type=int, default=1000)
@click.option("--save-total-limit", type=int, default=10)
@click.option("--local-output-dir", type=str, default="output/llama_zh001")
@click.option("--warmup-steps", type=int, default=1000)
@click.option("--max-source-length", type=int, default=256)
@click.option("--max-target-length", type=int, default=1024)
@click.option("--gradient-accumulation-steps", type=int, default=8)
def main(**kwargs):
    train(**kwargs)


if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )
    main()
