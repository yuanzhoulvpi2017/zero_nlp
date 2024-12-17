from embedding.data import ImageTextDataset, ImageTextDataCollator
from embedding.model import TextModelEmbedding, ImageModelEmbedding
from pathlib import Path
import torch
from PIL import Image
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from typing import List
from pymilvus import MilvusClient
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from dataclasses import dataclass


st.set_page_config(
    page_icon="🔍",
    page_title="mivlus 图文 Search",
    layout="wide",
)


@dataclass
class MilvusConfig:
    DB_PATH: str = "milvus_data/milvus_data.db"
    TABLE_NAME: str = "image_match_1216"
    DIM_VALUE: str = 1024


milvus_config = MilvusConfig()


@st.cache_resource
def load_text_embedding_model():
    text_embedding_model_path = "models/BAAI/bge-large-zh-v1.5"

    device = "cuda:0"
    text_embedding_model = TextModelEmbedding(text_embedding_model_path, device)
    return text_embedding_model


@st.cache_resource
def load_milvus_engine():
    client = MilvusClient(uri=milvus_config.DB_PATH)

    return client


def plot_images_with_scores(images, scores, rows=1, cols=None):
    """
    展示多个图片并显示对应的分数

    参数:
    - images: 图片文件路径列表 (List[Path])
    - scores: 对应的分数列表
    - rows: 行数（默认为1）
    - cols: 列数（默认为None，将自动计算）
    """
    # 如果没有指定列数，自动计算
    if cols is None:
        cols = len(images)

    # 创建图形和子图
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))

    # 如果只有一张图，将axes转换为数组
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    # 遍历并显示图片
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < len(images):
                # 读取图片
                img = mpimg.imread(str(images[idx]))

                # 显示图片
                axes[i, j].imshow(img)
                axes[i, j].axis("off")

                # 添加分数标签
                axes[i, j].set_title(f"Score: {scores[idx]:.4f}", fontsize=20)

    # 调整布局
    plt.tight_layout()
    plt.show()
    return fig


text_embedding_model = load_text_embedding_model()
milvus_client = load_milvus_engine()


def convert_text2embedding(text: str, normalize: bool = True):
    with torch.inference_mode():
        text_embedding = (
            text_embedding_model(
                encoded_input=None, sentences=[text], normalize_embeddings=normalize
            )
            .cpu()
            .numpy()
        )
        return text_embedding


def func1():
    with st.spinner("正在搜索中..."):
        with st.chat_message("user"):
            st.markdown(f"搜索关键词: {st.session_state.get('search_text')}")

        row_value, col_value = (
            st.session_state.get("row1"),
            st.session_state.get("col1"),
        )
        search_text = st.session_state.get("search_text")
        text_embedding = convert_text2embedding(search_text)
        res = milvus_client.search(
            collection_name=milvus_config.TABLE_NAME,
            data=[text_embedding.flatten().tolist()],
            limit=row_value * col_value,
            output_fields=["id", "text"],
        )[0]

        test_df = pd.DataFrame.from_dict(res).pipe(
            lambda x: x.assign(
                **{"image_path": x["entity"].apply(lambda j: j.get("text"))}
            )
        )

        image_paths = test_df["image_path"].tolist()

        scores = test_df["distance"].tolist()

        fig = plot_images_with_scores(
            image_paths,
            scores,
            rows=col_value,
            cols=row_value,
        )
        st.session_state["search_data"] = test_df

        with st.chat_message("ai"):
            tab1, tab2 = st.tabs(["图片", "表格"])
            with tab1:
                st.pyplot(fig)

            with tab2:
                st.table(
                    st.session_state["search_data"][["id", "distance", "image_path"]]
                )


with st.sidebar:
    st.title("图文搜索")
    st.markdown(
        "[关注b站: 良睦路程序员](https://space.bilibili.com/45156039?spm_id_from=333.1007.0.0)"
    )
    st.number_input(
        label="显示多少列图片", value=3, min_value=2, max_value=6, step=1, key="col1"
    )
    st.number_input(
        label="显示多少行图片", value=5, min_value=2, max_value=10, step=1, key="row1"
    )


st.chat_input("请输入搜索关键词：", key="search_text", on_submit=func1)
