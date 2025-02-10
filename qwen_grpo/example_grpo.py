# train_grpo.py
from datasets import load_dataset
from peft import LoraConfig
from trl_main import GRPOConfig, GRPOTrainer
import torch

# Load the dataset
dataset = load_dataset("data/trl-lib/tldr", split="train")

training_args = GRPOConfig(
    output_dir="output/Qwen2-0.5B-GRPO",
    learning_rate=1e-4,
    logging_steps=10,
    gradient_accumulation_steps=2,
    max_completion_length=128,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=2,
    model_init_kwargs={"torch_dtype": torch.bfloat16},
    num_train_epochs=1,
    save_steps=1000,
)
trainer = GRPOTrainer(
    model="model/Qwen2.5-0.5B-Instruct",
    reward_funcs="model/weqweasdas/RM-Gemma-2B",
    args=training_args,
    train_dataset=dataset,
    peft_config=LoraConfig(task_type="CAUSAL_LM"),
)

trainer.train()
trainer.save_model(training_args.output_dir)
