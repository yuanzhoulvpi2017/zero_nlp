from transformers.models.qwen2 import Qwen2ForCausalLM, Qwen2Tokenizer
import torch


model_name_or_path = "/home/yuanz/documents/models/Qwen/Qwen2.5-0.5B"
model = Qwen2ForCausalLM.from_pretrained(
    model_name_or_path, device_map="cuda:0", _attn_implementation="eager"
)
tokenizer = Qwen2Tokenizer.from_pretrained(model_name_or_path)


# demo 2
# 1. 直接基于第一步，生成新token

model_inputs1 = {
    "input_ids": torch.tensor(
        [[109432, 104130, 9370, 99584, 103852, 45995]], dtype=torch.long
    ).to(model.device),
    "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1]], dtype=torch.long).to(
        model.device
    ),
}

model_outputs1 = model.forward(
    **model_inputs1, use_cache=True
)  # model(**model_inputs1)
model_outputs1.keys()


# demo 4

# 1. 直接模拟简单粗暴类型的生成方式

print("demo4")

model_inputs3 = {
    "input_ids": torch.tensor(
        [[109432, 104130, 9370, 99584, 103852, 45995, 3837]], dtype=torch.long
    ).to(model.device),
    "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 1]], dtype=torch.long).to(
        model.device
    ),
}

model_inputs3


model_outputs3 = model.forward(**model_inputs3)  # model(**model_inputs3)
model_outputs3.keys()


# demo 3
# 1. 把上一次生成的past kv 拿过来，加上新拼接的token，生成
print("demo3")

model_outputs2 = model.forward(
    **{
        "input_ids": torch.tensor([[3837]], dtype=torch.long).to(model.device),
        "attention_mask": torch.tensor([[1]], dtype=torch.long).to(model.device),
    },
    past_key_values=model_outputs1.past_key_values,
)
model_outputs2.keys()
