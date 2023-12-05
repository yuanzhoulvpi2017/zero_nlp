# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Note: you need to install transformers from main to run this script. See https://huggingface.co/docs/transformers/installation#install-from-source
# TODO: bump transformers version in requirements at next release.

# 0. imports
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from peft import LoraConfig, get_peft_model

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments
from transformers.generation.utils import GenerationConfig

from trl import DPOTrainer


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    # data parameters
    beta: Optional[float] = field(
        default=0.1, metadata={"help": "the beta parameter for DPO loss"})

    # traindata parameters
    train_data: Optional[str] = field(
        default="/data2/huzheng/train_dpo/data/hh-rlhf", metadata={"help": "训练数据的位置"})

    # training parameters
    model_name_or_path: Optional[str] = field(
        default="gpt2", metadata={"help": "the model name"})
    # learning_rate: Optional[float] = field(
    #     default=1e-3, metadata={"help": "optimizer learning rate"})
    # per_device_train_batch_size: Optional[int] = field(
    #     default=4, metadata={"help": "batch size per device"})
    # gradient_accumulation_steps: Optional[int] = field(
    #     default=1, metadata={"help": "the number of gradient accumulation steps"}
    # )
    max_length: Optional[int] = field(
        default=512, metadata={"help": "max length of each sample"})
    max_prompt_length: Optional[int] = field(
        default=128, metadata={"help": "max length of each sample's prompt"})
    max_target_length: Optional[int] = field(
        default=128, metadata={"help": "Only used for encoder decoder model. Max target of each sample's prompt"}
    )
    label_pad_token_id: Optional[int] = field(
        default=-100, metadata={"help": "label for non response tokens"})
    # max_steps: Optional[int] = field(
    #     default=1000, metadata={"help": "max number of training steps"})
    # instrumentation
    # sanity_check: Optional[bool] = field(
    #     default=True, metadata={"help": "only train on 1000 samples"})
    # report_to: Optional[str] = field(
    #     default=None,
    #     metadata={
    #         "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
    #         '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
    #         'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
    #     },
    # )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    # gradient_checkpointing: Optional[bool] = field(
    #     default=False, metadata={"help": "Whether to use gradient checkpointing or no"}
    # )
    # gradient_checkpointing_kwargs: Optional[dict] = field(
    #     default=None,
    #     metadata={
    #         "help": "key word arguments to be passed along `torch.utils.checkpoint.checkpoint` method - e.g. `use_reentrant=False`"
    #     },
    # )
    # save_strategy: Optional[str] = field(
    #     default="steps"
    # )
    # save_steps: Optional[int] = field(
    #     default=100
    # )


def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = "\n\nAssistant:"
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != - \
        1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[: search_term_idx + len(search_term)]


def get_hh(train_data_path: str, silent: bool = False, cache_dir: str = None) -> Tuple[Dataset, Dataset]:
    """Load the Anthropic Helpful-Harmless dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts should be structured as follows:
      \n\nHuman: <prompt>\n\nAssistant:
    Multiple turns are allowed, but the prompt should always start with \n\nHuman: and end with \n\nAssistant:.
    """
    datasetall = load_dataset(
        "json",
        data_files={
            "train": [f"{train_data_path}/test_hhrlhf.json"],
            "test": [f"{train_data_path}/train_hhrlhf.json"],
        },
        cache_dir=cache_dir,
    )

    def split_prompt_and_responses(sample) -> Dict[str, str]:
        prompt = extract_anthropic_prompt(sample["chosen"])
        return {
            "prompt": prompt.replace("\n\nHuman: ", "<reserved_106>").replace(
                "\n\nAssistant: ", "<reserved_107>"),
            "chosen": sample["chosen"][len(prompt):],
            "rejected": sample["rejected"][len(prompt):],
        }
    datasetall = datasetall.map(split_prompt_and_responses)
    dataset_train = datasetall['test']
    dataset_test = datasetall['train']

    return dataset_train, dataset_test


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()  # [0]

    # 1. load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype='auto',
        # device_map='auto'
    )
    model.generation_config = GenerationConfig.from_pretrained(
        script_args.model_name_or_path)

    # 1.1 laod peft model
    LORA_R = 32
    # LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    TARGET_MODULES = [
        "o_proj", "W_pack",
    ]

    config = LoraConfig(
        r=LORA_R,
        # lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    # model.to('cuda')

    # model_ref = AutoModelForCausalLM.from_pretrained(
    #     script_args.model_name_or_path, trust_remote_code=True, torch_dtype='auto',
    #     device_map='auto'
    # )
    model_ref = None

    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with training_args.main_process_first(desc="loading and tokenization"):
        # 2. Load train and Load evaluation dataset
        train_dataset, eval_dataset = get_hh(
            train_data_path=script_args.train_data)

    # 4. initialize training arguments:
    # training_args = TrainingArguments(
    #     per_device_train_batch_size=script_args.per_device_train_batch_size,
    #     max_steps=script_args.max_steps,
    #     remove_unused_columns=False,
    #     gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    #     learning_rate=script_args.learning_rate,
    #     evaluation_strategy="steps",
    #     logging_first_step=True,
    #     logging_steps=10,  # match results in blog post
    #     eval_steps=500,
    #     output_dir="./test",
    #     optim="rmsprop",
    #     warmup_steps=150,
    #     report_to=script_args.report_to,
    #     bf16=True,
    #     save_strategy=script_args.save_strategy,
    #     save_steps=script_args.save_steps,
    #     gradient_checkpointing=script_args.gradient_checkpointing,
    #     # TODO: uncomment that on the next transformers release
    #     # gradient_checkpointing_kwargs=script_args.gradient_checkpointing_kwargs,
    # )

    # 5. initialize the DPO trainer
    dpo_trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=script_args.max_length,
        max_target_length=script_args.max_target_length,
        max_prompt_length=script_args.max_prompt_length,
        generate_during_eval=True,
    )

    # 6. train
    dpo_trainer.train()
