# %%
from MyTrainer import Trainer
from transformers import TrainingArguments
from transformers import DataCollatorForLanguageModeling
import random
from glob import glob
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModel
from peft import get_peft_model, LoraConfig, TaskType

tokenizer = AutoTokenizer.from_pretrained("thuglm", trust_remote_code=True)
model = AutoModel.from_pretrained(
    "thuglm", trust_remote_code=True).half().cuda()


peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1,
    # ['dense','dense_h_to_4h','dense_4h_to_h'] # 'query_key_value',
    target_modules=['dense',
                    'dense_h_to_4h', 'dense_4h_to_h'], )
model = get_peft_model(model, peft_config)
# %%
# import torch
# model = torch.quantization.quantize_dynamic(
#     model, {torch.nn.Linear}, dtype=torch.qint8
# )
# print(model)


# %%
random.seed(42)

all_file_list = glob(pathname="D:/数据/train_gpt2/gpt2_data/*/**")[:10]
test_file_list = random.sample(all_file_list, 3)
train_file_list = [i for i in all_file_list if i not in test_file_list]

len(train_file_list), len(test_file_list)

# %%
raw_datasets = load_dataset("csv", data_files={
                            'train': train_file_list, 'valid': test_file_list}, cache_dir="cache_data")

raw_datasets

# %%
context_length = 32


def tokenize(element):
    outputs = tokenizer(
        element["content"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}


tokenized_datasets = raw_datasets.map(
    tokenize, batched=True, remove_columns=raw_datasets["train"].column_names
)
tokenized_datasets

# %%
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# %%

args = TrainingArguments(
    output_dir="test001",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    evaluation_strategy="steps",
    eval_steps=2_0,
    logging_steps=2_0,
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    weight_decay=0.1,
    warmup_steps=1_000,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    save_steps=2_0,
    fp16=True,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
)
trainer.train()

# %%
