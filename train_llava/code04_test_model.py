import requests
from PIL import Image

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, LlavaProcessor

# model_id = "test_model_copy/model001"
model_id = "show_model/model001"  #


model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(0)

processor = LlavaProcessor.from_pretrained(model_id)

prompt_text = "<image>\nWhat are these?"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt_text},
]
prompt = processor.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_file = "000000039769.jpg"

raw_image = Image.open(image_file)
inputs = processor(prompt, raw_image, return_tensors="pt").to(0, torch.float16)


output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print(processor.decode(output[0], skip_special_tokens=False))
