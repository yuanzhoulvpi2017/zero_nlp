


# ChatGLM-6B-PT-Parallel
1. 本仓库基于[P-Tuning v2](https://github.com/THUDM/P-tuning-v2)代码，实现了**模型并行**。
2. 大部分文件包括本`readme.md`内容，都是来自于官网的代码。
3. 我只是做了网络层的设备映射，修改了部分代码。比如`modeling_chatglm.py`、`main_parallel.py`


下面以 [ADGEN](https://aclanthology.org/D19-1321.pdf) (广告生成) 数据集为例介绍代码的使用方法。

## 软件依赖
运行微调需要4.27.1版本的`transformers`。除 ChatGLM-6B 的依赖之外，还需要安装以下依赖
```
pip install rouge_chinese nltk jieba datasets
```
## 使用方法

### 下载数据集
ADGEN 数据集任务为根据输入（content）生成一段广告词（summary）。

```json
{
    "content": "类型#上衣*版型#宽松*版型#显瘦*图案#线条*衣样式#衬衫*衣袖型#泡泡袖*衣款式#抽绳",
    "summary": "这件衬衫的款式非常的宽松，利落的线条可以很好的隐藏身材上的小缺点，穿在身上有着很好的显瘦效果。领口装饰了一个可爱的抽绳，漂亮的绳结展现出了十足的个性，配合时尚的泡泡袖型，尽显女性甜美可爱的气息。"
}
```

从 [Google Drive](https://drive.google.com/file/d/13_vf0xRTQsyneRKdD1bZIr93vBGOczrk/view?usp=sharing) 或者 [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/b3f119a008264b1cabd1/?dl=1) 下载处理好的 ADGEN 数据集，将解压后的 `AdvertiseGen` 目录放到本目录下。

### 训练
运行以下指令进行训练：
```shell
bash train_parallel.sh
```
1. 注意，在`main_parallel.py`文件中，有个`device_map_dict`变量，这个是用来做网络层映射的。默认是两个`GPU`，可以自己修改。

### 推理

使用`infer_ptuning.ipynb`文件进行推理



