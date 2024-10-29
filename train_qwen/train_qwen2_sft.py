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

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="model/Qwen1.5-4B-Chat")
    use_lora: Optional[bool] = field(default=False)


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    source_length: int = field(default=128)
    target_length: int = field(default=512)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_deepspeed: bool = field(default=False)


def get_all_datapath(dir_name: str) -> List[str]:
    all_file_list = []
    # all_file_size = []

    for root, dir, file_name in os.walk(dir_name):
        for temp_file in file_name:
            standard_path = f"{root}/{temp_file}"

            all_file_list.append(standard_path)

    return all_file_list


def load_dataset_from_path(
    data_path: Optional[str] = None, cache_dir: Optional[str] = "cache_data"
) -> Dataset:
    all_file_list = get_all_datapath(data_path)  # [:1]
    data_files = {"train": all_file_list}
    extension = all_file_list[0].split(".")[-1]

    logger.info("load files %d number", len(all_file_list))

    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=cache_dir,
    )["train"]
    return raw_datasets


IGNORE_INDEX = -100


def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    ne_pad_token_id = (
        IGNORE_INDEX if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    )
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(ne_pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def find_subsequence(main_seq: List[int], sub_seq: List[int]) -> List[int]:
    positions = []
    for i in range(len(main_seq) - len(sub_seq) + 1):
        if main_seq[i : i + len(sub_seq)] == sub_seq:
            positions.append(i)
    return positions


def make_train_dataset(
    tokenizer: transformers.PreTrainedTokenizer,
    data_path: str,
    data_args: DataArguments,
) -> Dataset:
    logging.warning("Loading data...")

    dataset = load_dataset_from_path(
        data_path=data_path,
    )
    logging.warning("Formatting inputs...")

    def generate_sources_targets(
        examples: Dict, tokenizer: transformers.PreTrainedTokenizer
    ):
        ins_data = examples["instruction"]
        output = examples["output"]

        prompt = f"患者描述的内容：\n\n\n {ins_data}"
        messages = [
            {
                "role": "system",
                "content": "你是一个非常厉害的医生，精通各种医术，现在有患者开始向你描述他的情况，请帮帮他",
            },
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": output},
        ]
        token_id_list = tokenizer.apply_chat_template(messages)
        sub_sequence = [77091, 198]
        last_gen_id = find_subsequence(token_id_list, sub_sequence)[-1] + 2

        input_ids = token_id_list.copy()

        labels = [-100] * last_gen_id + token_id_list[
            (last_gen_id - len(token_id_list)) :
        ]

        examples["input_ids"] = input_ids
        examples["labels"] = labels
        return examples

    generate_sources_targets_p = partial(generate_sources_targets, tokenizer=tokenizer)

    dataset = dataset.map(
        function=generate_sources_targets_p,
        batched=False,
        desc="Running tokenizer on train dataset",
        num_proc=4,
    ).shuffle()
    return dataset


def load_model_and_tokenizer(
    model_args: ModelArguments,
    training_args: TrainingArguments,
) -> tuple:
    if training_args.use_deepspeed:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            torch_dtype="auto",
            trust_remote_code=True,
        )
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
        )

    if model_args.use_lora:
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

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True
    )

    return model, tokenizer


class HzTrainer(Trainer):
    def _maybe_log_save_evaluate(
        self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval
    ):
        if (
            self.control.should_log
            and self.state.global_step > self._globalstep_last_logged
        ):
            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(
                tr_loss_scalar
                / (self.state.global_step - self._globalstep_last_logged),
                4,
            )
            if grad_norm is not None:
                logs["grad_norm"] = (
                    grad_norm
                    if not isinstance(grad_norm, torch.Tensor)
                    else grad_norm.detach().item()
                )
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

            # Run delayed LR scheduler now that metrics are populated
            if isinstance(
                self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                self.lr_scheduler.step(metrics[metric_to_check])

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(
                self.args, self.state, self.control
            )


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model, tokenizer = load_model_and_tokenizer(model_args, training_args)

    with training_args.main_process_first(desc="loading and tokenization"):
        train_dataset = make_train_dataset(
            tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args
        )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, label_pad_token_id=IGNORE_INDEX
    )

    trainer = HzTrainer(
        model=model,
        tokenizer=tokenizer,
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
