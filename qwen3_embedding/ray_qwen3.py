import ray
import torch
from ray import serve
from typing import List, Dict
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ray.serve.config import HTTPOptions
import time
from test_qwen3_embedding import Qwen3Embedding
from test_qwen3_reranker import Qwen3Reranker

model_name_or_path_reranker = "models/Qwen3-Reranker-0.6B"
model_name_or_path_embedding = "models/Qwen3-Embedding-0.6B"


NUM_REPLICAS = 2  # 这个表示你需要这个进程启动几份。
NUM_GPUS = 0.9  # 这个表示，每一份占用多少个gpu （这种小模型，不需要设置超过1）

# 当前参数表示，会启动2份，第1，2，3会在显卡1上。第4，5，6会在显卡2上。


## example 2
# NUM_REPLICAS = 6
# NUM_GPUS = 0.5
# 当前参数表示，会启动6份，第1，2会在显卡1上。第3，4会在显卡2上。5，6会在显卡3上。

## example 3
# NUM_REPLICAS = 2
# NUM_GPUS = 0.6
# 当前参数表示，会启动2份，第1会在显卡1上。第2会在显卡2上


class RerankerInput(BaseModel):
    questions: List[str]
    texts: List[str]


class EmbeddingInput(BaseModel):
    input: list[str]
    is_query: bool


app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@serve.deployment(num_replicas=NUM_REPLICAS, ray_actor_options={"num_gpus": NUM_GPUS})
@serve.ingress(app)
class BatchCombineInferModel:
    def __init__(
        self, model_name_or_path_reranker: str, model_name_or_path_embedding: str
    ):
        self.emodel_embedding = Qwen3Embedding(
            model_name_or_path=model_name_or_path_embedding,
        )
        self.model_reranker = Qwen3Reranker(
            model_name_or_path=model_name_or_path_reranker,
            instruction="Retrieval document that can answer user's query",
            max_length=2048,
        )

    @app.post("/embedding/api")
    def embedding(self, texts: EmbeddingInput):
        with torch.inference_mode():
            output = self.emodel_embedding.encode(texts.input, is_query=texts.is_query)
            return output.cpu().detach().numpy().tolist()

    @app.post("/reranker/api")
    def reranker(self, texts: RerankerInput):
        with torch.inference_mode():
            pairs = list(zip(texts.questions, texts.texts))

            instruction = "Given the user query, retrieval the relevant passages"
            new_scores = self.model_reranker.compute_scores(pairs, instruction)
            return new_scores


serve.start(http_options=HTTPOptions(host="0.0.0.0", port=4008))


serve.run(
    BatchCombineInferModel.bind(
        model_name_or_path_reranker, model_name_or_path_embedding
    ),
    route_prefix="/",
)


while True:
    time.sleep(1000)
