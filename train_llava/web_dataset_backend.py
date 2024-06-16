from typing import Dict, List, Any

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from train_llava.data_websend import SendDatasetByWeb


webdatasetsend = SendDatasetByWeb(
    model_name_or_path="test_model/model001",
    dataset_dir="data/liuhaotian/LLaVA-CC3M-Pretrain-595K",
    cache_dir="data/cache_data",
    num_proc=10
)


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/len")
async def get_len():
    return len(webdatasetsend)


@app.get("/slice")
async def get_slice(index: int) -> dict[Any, Any]:
    return webdatasetsend[index]


if __name__ == "__main__":
    uvicorn.run(app=app, host="0.0.0.0", port=7001, reload=False)
