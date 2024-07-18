import streamlit as st
import requests
from transformers import LlavaForConditionalGeneration, AutoProcessor
import torch
from PIL import Image
import numpy as np

st.set_page_config(page_title="demo4llava", page_icon="🧑‍💻", layout="wide")


@st.cache_resource
def load_llava_model(raw_model_name_or_path: str):

    # raw_model_name_or_path = (
    #     "output_model_freeze_vison_0705"  # "output_model_lora_merge_001"
    # )
    model = LlavaForConditionalGeneration.from_pretrained(
        raw_model_name_or_path,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.bfloat16,
    )

    processor = AutoProcessor.from_pretrained(raw_model_name_or_path)
    model.eval()
    print("ok")
    return model, processor


def build_model_input(model, processor, testdata: tuple):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": testdata[0]},
    ]
    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # print(prompt)
    # print("*"*20)
    image = Image.open(testdata[2])
    inputs = processor(text=prompt, images=image, return_tensors="pt")

    for tk in inputs.keys():
        inputs[tk] = inputs[tk].to(model.device)
    generate_ids = model.generate(**inputs, max_new_tokens=1024)

    generate_ids = [
        oid[len(iids) :] for oid, iids in zip(generate_ids, inputs.input_ids)
    ]

    gen_text = processor.batch_decode(
        generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )[0]
    return gen_text


llava_model, llava_processor = load_llava_model(
    "output_model_lora_merge_gpt4oocrshared"
)


def func1():
    image_file_path = st.session_state.get("input_data_002", None)
    user_input = st.session_state.get("input_data_001", "")
    if (len(user_input) >= 4) and ("<image>" in user_input):
        if image_file_path is not None:
            test_data = (user_input, None, image_file_path)
            with st.spinner("正在生产文本...请稍等"):
                model_gen_text = build_model_input(
                    llava_model, llava_processor, test_data
                )
                st.session_state["output_data1"] = model_gen_text

        else:
            st.warning("需要输入图像")
    else:
        st.warning(
            "必须输入有效文本,比如: `<image>\nPlease describe specifically what you observed in the picture`"
        )


with st.sidebar:
    st.text_area(
        label="输入文本",
        value="<image>\nPlease describe specifically what you observed in the picture",
        key="input_data_001",
    )
    st.file_uploader("上传一张照片", type=["png", "jpg"], key="input_data_002")
    st.button("生成", on_click=func1)


st.markdown("### 展示模型生成的文本")
if len(st.session_state.get("output_data1", "")) > 0:
    st.markdown(st.session_state.get("output_data1", ""))

st.markdown("### 展示上传的图像")

if st.session_state.get("input_data_002", None) is not None:
    image = Image.open(st.session_state.get("input_data_002", None))
    st.image(image)


# build_model_input(llava_model, llava_model, testdata)
