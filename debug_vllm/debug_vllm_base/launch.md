

```bash
python -m vllm.entrypoints.openai.api_server --served-model-name Qwen1.5-7B-Chat --model /home/yuanz/documents/train_qwen/model/Qwen1.5-1.8B-Chat --tensor-parallel-size 2 --port 8001
```
[
                "--served-model-name",
                "Qwen1.5-7B-Chat",
                "--model",
                "/home/yuanz/documents/train_qwen/model/Qwen1.5-1.8B-Chat",
                "--tensor-parallel-size",
                "2",
                "--port",
                "8001"
            ]

```python

from openai import OpenAI
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8001/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

chat_response = client.chat.completions.create(
    model="Qwen1.5-7B-Chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me something about large language models."},
    ]
)
print("Chat response:", chat_response)
```