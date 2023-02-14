import gradio as gr
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer

model_name_or_path = "checkpoint-36000"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

model = GPT2LMHeadModel.from_pretrained(model_name_or_path, pad_token_id=tokenizer.eos_token_id)


def infer_model(txt: str) -> str:
    # encode context the generation is conditioned on
    input_ids = tokenizer.encode(txt, return_tensors='pt')
    # set no_repeat_ngram_size to 2
    beam_output = model.generate(
        input_ids,
        max_length=200,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True
    )

    result = tokenizer.decode(beam_output[0], skip_special_tokens=True)
    result = result.replace(" ", "")
    return result


def predict(input, history=[]):
    respose = infer_model(input)
    res = [(input, respose)]
    return res, history


def add_text(state, text):
    res = infer_model(text)
    state = state + [(text, res)]
    return state, state


with gr.Blocks(css="#chatbot .overflow-y-auto{height:500px}") as demo:
    chatbot = gr.Chatbot(elem_id="chatbot")
    state = gr.State([])

    with gr.Row():
        txt = gr.Textbox(show_label=False, placeholder="输入文本").style(container=False)

    txt.submit(add_text, [state, txt], [state, chatbot])

if __name__ == "__main__":
    demo.launch(server_name='0.0.0.0')
