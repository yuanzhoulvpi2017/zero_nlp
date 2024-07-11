# 训练llava

1. 模型构建：基于`openai/clip-vit-large-patch14-336` 和`Qwen1.5-4B-Chat`模型，构建一个llava模型
2. 数据构建：`liuhaotian/LLaVA-CC3M-Pretrain-595K`
3. 训练方式：基于`deepspeed-zero2`，有`lora`训练、全量参数训练、冻结视觉层进行训练等方式。

## 具体教程

| 任务流程          | 细节                                                        | 关联代码                                                                                                                                                                                                                                                                    | 关联视频                                                                                                                  |
|---------------|-----------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| 认识llava模型     | 从transformers源码角度、带你图解llava                               | /                                                                                                                                                                                                                                                                       | [B站: 多模态大模型LLaVA模型讲解——transformers源码解读](https://www.bilibili.com/video/BV1nw4m1S7nZ/?spm_id_from=333.999.0.0)         |
| 从0到1构建llava模型 | 1. 如何从0到1构建一个空的属于自己的llava模型<br/>2. 加深对llava模型的认识，为训练模型做铺垫 | [code03_build_model_show.ipynb](https://github.com/yuanzhoulvpi2017/zero_nlp/blob/main/train_llava/code03_build_model_show.ipynb)                                                                                                                                       | [B站: 自定义多模态大模型LLaVA——LLaVA系列](https://www.bilibili.com/video/BV1GS411P74b/?spm_id_from=333.999.0.0)                   |
| 构建训练数据集       | 如何基于`liuhaotian/LLaVA-CC3M-Pretrain-595K`数据集，构建训练数据集      | [train_llava/train_llava/data.py](https://github.com/yuanzhoulvpi2017/zero_nlp/blob/main/train_llava/train_llava/data.py)                                                                                                                                               | [B站：训练LLaVA模型（数据集构建、基于Trainer的训练框架搭建）——LLaVA系列](https://www.bilibili.com/video/BV1Si421v7j1/?spm_id_from=333.999.0.0) |
| 训练流程          | 1. 基于transformers框架，搭建训练代码<br/>2. 实现多重模式的训练。              | [train_llava/run_zero2.sh](https://github.com/yuanzhoulvpi2017/zero_nlp/blob/main/train_llava/run_zero2.sh)                                                                                                                                                             | [B站：训练LLaVA模型（数据集构建、基于Trainer的训练框架搭建）——LLaVA系列](https://www.bilibili.com/video/BV1Si421v7j1/?spm_id_from=333.999.0.0) |
| 花式训练          | 1. 分享训练训练技巧                                               | todo                                                                                                                                                                                                                                                                    |                                                                                                                       |
| 推理            | 训练的模型，如何进行推理                                              | 1. lora版本： [code05_infer_lora.ipynb](https://github.com/yuanzhoulvpi2017/zero_nlp/blob/main/train_llava/code05_infer_lora.ipynb) <br/>2. 全量参数版本:[train_llava/code05_infer.ipynb](https://github.com/yuanzhoulvpi2017/zero_nlp/blob/main/train_llava/code05_infer.ipynb) |                                                                                                                       |

## 下载模型

1.

hf的链接为：[llava_qwen15-4b-chat_openai-clip-vit-large-patch14-336](https://huggingface.co/yuanzhoulvpi/llava_qwen15-4b-chat_openai-clip-vit-large-patch14-336)

## 训练策略

| 训练方式                         | 视觉层  | 转接层  | 语言层        | 效果评估     |
|------------------------------|------|------|------------|----------|
| `--train_type use_lora`      | 冻结🧊 | 训练🔥 | 训练🔥（部分参数） | 效果非常好 👍 |
| `--train_type none`          | 训练🔥 | 训练🔥 | 训练🔥       | 效果非常差👎  |
| `--train_type freeze_vision` | 冻结🧊 | 训练🔥 | 训练🔥（全量参数  | 尚未评估     |

1. 训练的时候，使用lora方式进行训练最好。在`run_zero2.sh`里面设置`--train_type use_lora`即可。
2. 全量参数训练，效果非常差。

## 训练技巧

为了可以异步的处理数据，可以在`run_zero2.sh`里面使用这三个参数

```shell
    --dataloader_pin_memory True \
    --dataloader_num_workers 10 \
    --dataloader_persistent_workers True \

```

基本上可以提高1倍的训练效率。
参考链接：

1. https://developer.aliyun.com/article/914214
2. https://blog.csdn.net/qq_32527569/article/details/134777374