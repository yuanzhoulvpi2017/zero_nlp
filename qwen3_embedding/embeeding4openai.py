from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
# from src.custom_model import request_first_embedding
from pydantic import BaseModel
from typing import List, Union, Dict, Any
import random

# 定义请求和响应模型
class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str = "hhh"
    encoding_format: str = "float"
    user: str = None


# OpenAI API 返回格式的数据结构
class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[Dict[str, Any]]
    model: str
    usage: Dict[str, int]


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# 创建一个fake函数：将文本转换成向量
def request_first_embedding(text:str):
    return [random.random() for _ in range(10)]




# 定义API路由
@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def embeddings(request: EmbeddingRequest):
    try:
        # 确保输入是列表格式
        if isinstance(request.input, str):
            texts = [request.input]
        else:
            texts = request.input

        # 获取文本嵌入向量
        embeddings = [request_first_embedding(tt) for tt in texts]

        # 构造返回数据
        data = []
        total_tokens = 0

        for idx, embedding in enumerate(embeddings):
            # 转换numpy数组为普通列表并确保精度
            embedding_list = embedding + [0.0] * (128 - len(embedding))

            data.append(
                {"object": "embedding", "embedding": embedding_list, "index": idx}
            )

            # 简单的token计数估计
            total_tokens += len(texts[idx])

        response = {
            "object": "list",
            "data": data,
            "model": request.model,
            "usage": {"prompt_tokens": total_tokens, "total_tokens": total_tokens},
        }

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 运行应用
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=6002)
