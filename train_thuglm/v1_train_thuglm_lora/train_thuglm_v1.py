# %%
from train_thuglm.v1_train_thuglm_lora.thuglm.modeling_chatglm import ChatGLMForConditionalGeneration
from train_thuglm.v1_train_thuglm_lora.thuglm.tokenization_chatglm import ChatGLMTokenizer

import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_linear_schedule_with_warmup

from peft import LoraConfig, TaskType, get_peft_model
from peft.utils.other import fsdp_auto_wrap_policy
from tqdm import tqdm
from glob import glob

# %%
model = ChatGLMForConditionalGeneration.from_pretrained(
    "THUDM/chatglm-6b", load_in_8bit=False)

tokenizer = ChatGLMTokenizer.from_pretrained("THUDM/chatglm-6b")

# %%
accelerator = Accelerator()
batch_size = 8
text_column = "sentence"
label_column = "sentence"
max_length = 64
lr = 1e-3
num_epochs = 1

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1,
    # ['dense','dense_h_to_4h','dense_4h_to_h'] # 'query_key_value',
    target_modules=['dense',
                    'dense_h_to_4h', 'dense_4h_to_h'],
)
model = get_peft_model(model, peft_config)
accelerator.print(model.print_trainable_parameters())

# %%
all_data_list = glob("v1_train_thuglm_lora/data/*")[:10]

dataset = load_dataset(
    "csv",
    data_files={
        "train": all_data_list[:6],
        "validation": all_data_list[6:],
    },
)


# %%


# %%


def preprocess_function(examples):
    inputs = examples[text_column]
    targets = examples[label_column]
    model_inputs = tokenizer(
        inputs, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    labels = tokenizer(targets, max_length=64, padding="max_length",
                       truncation=True, return_tensors="pt")
    labels = labels["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["label"] = labels
    return model_inputs


# %%
# for i in tqdm(range(len(dataset['train']))):
#     try:
#         preprocess_function(dataset['train'][i])

#     except Exception as e:
#         print(i)
#         break


# %%
# dataset['train'][11]

# %%
# preprocess_function(dataset['train'][11])

# %%


processed_datasets = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=1,
    remove_columns=dataset["train"].column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)

train_dataset = processed_datasets["train"]
eval_dataset = processed_datasets["validation"]

# %%
train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
)
eval_dataloader = DataLoader(
    eval_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
)

# %%
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * num_epochs),
)

# %%
if getattr(accelerator.state, "fsdp_plugin", None) is not None:
    accelerator.state.fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(
        model)

model, train_dataloader, eval_dataloader, optimizer, lr_scheduler = accelerator.prepare(
    model, train_dataloader, eval_dataloader, optimizer, lr_scheduler
)
# model = model.half()
accelerator.print(model)

# %%
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        # print(batch.keys())
        # outputs = model(**batch)
        input_ids = batch["input_ids"].to(accelerator.device, dtype=torch.long)
        labels = batch["labels"].to(accelerator.device, dtype=torch.long)
        outputs = model(
            input_ids=input_ids,
            labels=labels,
        )
        loss = outputs.loss
        total_loss += loss.detach().float()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    model.eval()
    eval_loss = 0
    eval_preds = []
    for step, batch in enumerate(tqdm(eval_dataloader)):
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        eval_loss += loss.detach().float()
        preds = accelerator.gather_for_metrics(
            torch.argmax(outputs.logits, -1)).detach().cpu().numpy()
        eval_preds.extend(tokenizer.batch_decode(
            preds, skip_special_tokens=True))
    eval_epoch_loss = eval_loss / len(eval_dataloader)
    eval_ppl = torch.exp(eval_epoch_loss)
    train_epoch_loss = total_loss / len(train_dataloader)
    train_ppl = torch.exp(train_epoch_loss)
    accelerator.print(
        f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")

    correct = 0
    total = 0
    for pred, true in zip(eval_preds, dataset["validation"][label_column]):
        if pred.strip() == true.strip():
            correct += 1
        total += 1
    accuracy = correct / total * 100
    accelerator.print(f"{accuracy=}")
    accelerator.print(f"{eval_preds[:10]=}")
    accelerator.print(f"{dataset['validation'][label_column][:10]=}")
    # accelerator.wait_for_everyone()
    # model.push_to_hub(
    #     "smangrul/" + f"{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}".replace("/", "_"),
    #     state_dict=accelerator.get_state_dict(model),
    #     use_auth_token=True,
    # )
    accelerator.wait_for_everyone()

# %%
