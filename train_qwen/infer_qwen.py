import torch
from peft import peft_model,PeftModel
import transformers
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"


raw_model_name_or_path = "Qwen2-7B"
peft_model_name_or_path = "output_0309_1dot8b"
cache_dir = "cache_dir"

tokenizer = transformers.AutoTokenizer.from_pretrained(
        raw_model_name_or_path, trust_remote_code=True
    )


model = transformers.AutoModelForCausalLM.from_pretrained(
            raw_model_name_or_path,
            cache_dir=cache_dir,
            torch_dtype="auto",
            trust_remote_code=True,
        )
model = PeftModel.from_pretrained(model, peft_model_name_or_path, adapter_name="peft_v1")
model.eval()
print('ok')


model = model.merge_and_unload()
model.save_pretrained("output_model_lora_merge_001")
tokenizer.save_pretrained("output_model_lora_merge_001")