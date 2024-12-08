import json
from typing import Any, List

import requests
from openai import OpenAI
from tqdm.auto import tqdm
import os
import shutil
from pathlib import Path
from multiprocessing import Pool
from multiprocessing.pool import AsyncResult


def call_qwen_api(message: List[Any]) -> str:
    openai_api_key = "EMPTY"
    openai_api_base = "http://ip:port/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    chat_response = client.chat.completions.create(
        model="deepseek-2.5",  # "Qwen2.5-72B-Instruct",
        messages=message,
    )
    return chat_response.choices[0].message.content


def trans_data(prompt_en: str):
    message = [
        {
            "role": "user",
            "content": f"""
你是一位专业的中英文翻译，擅长准确、地道且自然的翻译。请将以下英文句子翻译成中文：

英文原文：{prompt_en}

翻译要求：
1. 保持原文的语义和语气
2. 使用流畅的中文表达
3. 避免直译和生硬的表达
4. 根据上下文选择最合适的翻译方式
""",
        },
    ]
    res = call_qwen_api(message)
    return res


def load_raw_data():
    with open("data/jackyhate/text-to-image-2M/data_2m_p1_large.json", "r") as fin:
        alldata = fin.readlines()

    alldata = [json.loads(i) for i in alldata]
    return alldata


def gen_data(data: List[Any], index: int, save_dir: Path, chunk_size: int):
    temp_file_path = str(save_dir / f"{index}.json")
    try:
        with open(temp_file_path, "r") as fin:
            total_len = len(fin.readlines())

        if total_len >= int(chunk_size * 0.9):
            need_regen = False
        else:
            need_regen = True
    except Exception as e:
        need_regen = True

    if need_regen:
        with open(temp_file_path, "w") as fout:
            for i in tqdm(data, desc=f"process {index}"):
                try:
                    trans_chinese = trans_data(i["prompt"])
                    fout.write(
                        json.dumps(
                            {"zh_prompt": trans_chinese, **i}, ensure_ascii=False
                        )
                        + "\n"
                    )
                except Exception as e:
                    print(e)


def func_call_pool(value: AsyncResult, p_bar: tqdm):
    p_bar.update()
    try:
        value.get(timeout=3600 * 24 * 10)
    except Exception as e:
        pass


def main(
    chunk_size: int = 5000, n_process: int = 10, save_dir: str = "data/temp_trans_data"
):
    save_dir = Path(save_dir)
    # shutil.rmtree(save_dir, ignore_errors=True)
    os.makedirs(save_dir, exist_ok=True)
    all_json_data = load_raw_data()
    all_json_data_chunk = [
        all_json_data[i : i + chunk_size]
        for i in range(0, len(all_json_data), chunk_size)
    ]

    p_bar = tqdm(total=len(all_json_data_chunk), desc="translating data")

    with Pool(processes=n_process) as P:
        pool_list = [
            P.apply_async(
                func=gen_data,
                args=(chunkdata, index, save_dir, chunk_size),
            )
            for index, chunkdata in enumerate(all_json_data_chunk)
        ]

        _ = [func_call_pool(p_, p_bar) for p_ in pool_list]


if __name__ == "__main__":
    main()
