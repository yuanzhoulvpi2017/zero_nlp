import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import torch
import transformers
from PIL import Image, ImageFile

from torchvision.io import ImageReadMode, read_image
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from transformers.utils import send_example_telemetry
from typing import List, Any

from embedding.model import MODEL_TRAIN_MAP
from embedding.data import ImageTextDataset, ImageTextData, ImageTextDataCollator
from embedding.trainer import HzTrainer
from embedding.utils import print_trainable_parameters

logger = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass
class ModelArguments:
    text_model_name_or_path: str
    image_model_name_or_path: str
    cross_embedding: str = field(
        default="base",
        metadata={
            "help": "cross embedding type: 'base' or 'custom' or 'gather_nn' or 'gather'"
        },
    )
    cache_dir: Optional[str] = field(default=None, metadata={"help": "cache_model_dir"})


@dataclass
class DataArguments:
    data_size: str = field(
        default="small", metadata={"help": "data size: 'small' or 'large'"}
    )
    language: str = field(default="en", metadata={"help": "language: 'en' or 'zh'"})
    max_seq_length: int = field(default=256, metadata={"help": "max sequence length"})


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.text_model_name_or_path, cache_dir=model_args.cache_dir
    )
    image_processor = AutoImageProcessor.from_pretrained(
        model_args.image_model_name_or_path, cache_dir=model_args.cache_dir
    )


    
    MODEL_TYPE = MODEL_TRAIN_MAP.get(model_args.cross_embedding, "base")
    logger.info(f"Training cross batch embedding type: {model_args.cross_embedding}")

    model = MODEL_TYPE(
        text_model_name_or_path=model_args.text_model_name_or_path,
        image_model_name_or_path=model_args.image_model_name_or_path,
        device="cuda",
    )
    set_seed(training_args.seed)

    imagetext_dataset = ImageTextDataset(
        data_size=data_args.data_size, language=data_args.language
    )
    data_collator = ImageTextDataCollator(
        image_size=model.image_model_embedding.model.config.image_size,
        mean=image_processor.image_mean,
        std=image_processor.image_std,
        tokenizer=tokenizer,
        max_seq_length=data_args.max_seq_length,
    )
    print_trainable_parameters(model)

    trainer = HzTrainer(
        model=model,
        args=training_args,
        train_dataset=imagetext_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )

    if training_args.do_train:
        checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        # tokenizer.save_pretrained(training_args.output_dir)
        # image_processor.save_pretrained(training_args.output_dir)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()


if __name__ == "__main__":
    main()
