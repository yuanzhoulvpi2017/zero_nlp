import logging
from dataclasses import dataclass

from order_train.data import TrainDatasetForOrder, GroupCollator

from order_train.model import MyModel
from order_train.trainer import MyTrainer

from transformers import TrainingArguments, HfArgumentParser


@dataclass
class DataAndModelArguments:
    dataset_dir: str
    cache_dir: str


def main():
    parser = HfArgumentParser((TrainingArguments, DataAndModelArguments))
    training_args, data_model_args = parser.parse_args_into_dataclasses()
    training_args: TrainingArguments
    data_model_args: DataAndModelArguments

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    model = MyModel()
    dataset = TrainDatasetForOrder(
        data_model_args.dataset_dir, data_model_args.cache_dir
    )

    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=GroupCollator(),
    )

    trainer.train()


if __name__ == "__main__":
    main()
