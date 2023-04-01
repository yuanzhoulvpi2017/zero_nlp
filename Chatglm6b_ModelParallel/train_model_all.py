# # 完整的训练流程
# 1. 数据基于`https://github.com/hikariming/alpaca_chinese_dataset`
# 2. 部分代码来源于`https://github.com/27182812/ChatGLM-chinese-insturct/blob/main/finetune.py`
# 3. 基于我之前修改的`model_chatglm.py`做的一整套教程
# 
# ## 清洗数据

# 在这里控制要使用的显卡
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# 如果没有下载这个仓库，可以使用下面命令进行clone

# !git clone https://github.com/hikariming/alpaca_chinese_dataset.git


########################################################################################################################
# part 1 数据准备
########################################################################################################################


from glob import glob
import os
import pandas as pd
import shutil
from itertools import chain
from tqdm import tqdm
from pathlib import Path

target_dir_list = ['alpaca_chinese_dataset/其他中文问题补充/',
                   'alpaca_chinese_dataset/翻译后的中文数据/',
                   'alpaca_chinese_dataset/chatglm问题数据补充/',
                   #    'alpaca_chinese_dataset/原始英文数据/'
                   ]

all_json_path = [glob(i + "*.json") for i in target_dir_list]
all_json_path = list(chain(*all_json_path))
len(all_json_path), all_json_path[:5]


def read_json(x: str):
    try:
        data = pd.read_json(x)
        return data
    except Exception as e:
        return pd.DataFrame()


alldata = pd.concat([read_json(i) for i in all_json_path])

genrate_data_dir = "data3_0328"
genrate_data_dir = Path(genrate_data_dir)

if genrate_data_dir.exists():
    shutil.rmtree(genrate_data_dir, ignore_errors=True)

os.makedirs(genrate_data_dir, exist_ok=True)

alldata = alldata.sample(frac=1).reset_index(drop=True)

chunk_size = 666

for index, start_id in tqdm(enumerate(range(0, alldata.shape[0], chunk_size))):
    temp_data = alldata.iloc[start_id:(start_id + chunk_size)]
    temp_data.to_csv(genrate_data_dir.joinpath(f"{index}.csv"), index=False)



########################################################################################################################
# part 2 模型加载和转换
########################################################################################################################



from thuglm.modeling_chatglm import ChatGLMForConditionalGeneration
from transformers import Trainer
from transformers import TrainingArguments
import random
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from peft import get_peft_model, LoraConfig, TaskType
from typing import Optional
import torch
from MyTrainer import Trainer

tokenizer = AutoTokenizer.from_pretrained("thuglm", trust_remote_code=True)

device_map_dict = {'transformer.word_embeddings': 0,
                   'transformer.layers.0': 0,
                   'transformer.layers.1': 0,
                   'transformer.layers.2': 0,
                   'transformer.layers.3': 0,
                   'transformer.layers.4': 0,
                   'transformer.layers.5': 0,
                   'transformer.layers.6': 0,
                   'transformer.layers.7': 0,
                   'transformer.layers.8': 0,
                   'transformer.layers.9': 0,
                   'transformer.layers.10': 0,
                   'transformer.layers.11': 0,
                   'transformer.layers.12': 0,
                   'transformer.layers.13': 0,
                   'transformer.layers.14': 0,
                   'transformer.layers.15': 1,
                   'transformer.layers.16': 1,
                   'transformer.layers.17': 1,
                   'transformer.layers.18': 1,
                   'transformer.layers.19': 1,
                   'transformer.layers.20': 1,
                   'transformer.layers.21': 1,
                   'transformer.layers.22': 1,
                   'transformer.layers.23': 1,
                   'transformer.layers.24': 1,
                   'transformer.layers.25': 1,
                   'transformer.layers.26': 1,
                   'transformer.layers.27': 1,
                   'transformer.final_layernorm': 1,
                   'lm_head': 1
                   }

model = AutoModel.from_pretrained(
    "thuglm", trust_remote_code=True).half().cuda()

for k, v in device_map_dict.items():
    if k == 'transformer.word_embeddings':
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(f'cuda:{v}')
    if k.find("transformer.layers") != -1:
        sub_value = int(k.replace("transformer.layers.", ""))
        model.transformer.layers[sub_value] = model.transformer.layers[sub_value].to(f'cuda:{v}')

    if k == "transformer.final_layernorm":
        model.transformer.final_layernorm = model.transformer.final_layernorm.to(f'cuda:{v}')

model.enable_input_require_grads()
torch.cuda.empty_cache()

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1,
    # ['dense','dense_h_to_4h','dense_4h_to_h'] # 'query_key_value',
    target_modules=['query_key_value', ],
)
model = get_peft_model(model, peft_config)

for k, v in device_map_dict.items():
    #     if k == 'transformer.word_embeddings':
    #         model.base_model.transformer.word_embeddings = model.base_model.transformer.word_embeddings.to(f'cuda:{v}')
    if k.find("transformer.layers") != -1:
        sub_value = int(k.replace("transformer.layers.", ""))
        model.base_model.transformer.layers[sub_value].attention.query_key_value.lora_A = \
            model.base_model.transformer.layers[sub_value].attention.query_key_value.lora_A.to(f'cuda:{v}')
        model.base_model.transformer.layers[sub_value].attention.query_key_value.lora_B = \
            model.base_model.transformer.layers[sub_value].attention.query_key_value.lora_B.to(f'cuda:{v}')

########################################################################################################################
# part 3 数据加载
########################################################################################################################


random.seed(42)

all_file_list = glob(pathname=genrate_data_dir.joinpath("*.csv").__str__())

test_file_list = random.sample(all_file_list, int(len(all_file_list) * 0.25))
train_file_list = [i for i in all_file_list if i not in test_file_list]
train_file_list, test_file_list = train_file_list[:5], test_file_list[:5]

len(train_file_list), len(test_file_list)

dataset = load_dataset(
    "csv",
    data_files={
        'train': train_file_list,
        'valid': test_file_list
    },
    cache_dir="cache_data"
)


def get_masks_and_position_ids(
        seq, seq_len, context_length, device, gmask=False, position_encoding_2d=True
):
    mask_position = (
            seq_len - 2
    )  # is equal to `seq.index(mask_token)` or `seq.index(150001)`
    attention_mask = torch.ones((1, context_length, context_length), device=device)
    attention_mask.tril_()
    attention_mask[..., : mask_position - 1] = 1
    attention_mask = (attention_mask < 0.5).bool()

    if position_encoding_2d:
        seq_length = seq_len - 1  # is equal to `seq_length = seq.index(150004)`
        position_ids = torch.arange(context_length, dtype=torch.long, device=device)
        if not gmask:
            position_ids[seq_length:] = mask_position
        block_position_ids = torch.cat(
            (
                torch.zeros(seq_length, dtype=torch.long, device=device),
                torch.arange(
                    context_length - seq_length, dtype=torch.long, device=device
                )
                + 1,
            )
        )
        position_ids = torch.stack((position_ids, block_position_ids), dim=0)
    else:
        position_ids = torch.arange(context_length, dtype=torch.long, device=device)
        if not gmask:
            position_ids[context_length - 1:] = mask_position
    return attention_mask, position_ids


def data_collator(features: list) -> dict:
    len_ids = [len(feature["input_ids"]) for feature in features]
    longest = max(len_ids) + 1
    input_ids = []
    attention_mask_list = []
    position_ids_list = []
    labels_list = []
    for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
        ids = feature["input_ids"]
        seq_len = feature["seq_len"]
        labels = (
                [-100] * (seq_len - 1)
                + ids[(seq_len - 1):]
                + [tokenizer.eos_token_id]
                + [-100] * (longest - ids_l - 1)
        )
        ids = ids + [tokenizer.eos_token_id] * (longest - ids_l)
        _ids = torch.LongTensor(ids)
        attention_mask, position_ids = get_masks_and_position_ids(
            ids, seq_len, longest, _ids.device, gmask=False
        )
        labels_list.append(torch.LongTensor(labels))
        input_ids.append(_ids)
        attention_mask_list.append(attention_mask)
        position_ids_list.append(position_ids)
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)
    attention_mask = torch.stack(attention_mask_list)
    position_ids = torch.stack(position_ids_list)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }


def format_example(example: dict) -> dict:
    context = f"Instruction: {example['instruction']}\n"
    if example.get("input"):
        context += f"Input: {example['input']}\n"
    context += "Answer: "
    target = example["output"]
    # {"context": context, "target": target}
    example['context'] = context
    example['target'] = target
    return example


max_seq_length = 1024


def preprocess(example):
    prompt = example["context"]
    target = example["target"]
    prompt_ids = tokenizer.encode(prompt, max_length=max_seq_length, truncation=True)
    target_ids = tokenizer.encode(
        target, max_length=max_seq_length, truncation=True, add_special_tokens=False
    )
    input_ids = prompt_ids + target_ids + [tokenizer.eos_token_id]
    return {"input_ids": input_ids, "seq_len": len(prompt_ids)}


def filter_nan(example):
    return example['target'] is not None and example['context'] is not None


tokenized_datasets = dataset.map(
    function=format_example, remove_columns=dataset['train'].column_names
).filter(function=filter_nan)
tokenized_datasets = tokenized_datasets.map(function=preprocess)


########################################################################################################################
# part 4 训练
########################################################################################################################


args = TrainingArguments(
    output_dir="modellog0040101",
    per_device_train_batch_size=4,  # 如果在24G显存上的显卡，可以开到4
    per_device_eval_batch_size=2,
    evaluation_strategy="steps",
    eval_steps=20,
    logging_steps=20,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    weight_decay=0.1,
    warmup_steps=1_000,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    save_steps=20,
    fp16=True,
    push_to_hub=False,
    remove_unused_columns=False
)

model.is_parallelizable = True
model.model_parallel = True

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
)
trainer.train()
