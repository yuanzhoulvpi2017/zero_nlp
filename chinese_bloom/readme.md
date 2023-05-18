# 🚀 使用`sft`数据训练中文`bloom`模型

## 📣 介绍
1. ✅ 基于[stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca)项目，使用`sft`格式数据对`bloom`模型微调；
2. ✅ 支持`deepspeed-zero2`、`deepspeed-zero3`；
3. ✅ 支持自定义数据，支持大数据训练；
4. ✅ 得益于`bloom`本身的能力，微调后的模型支持`中文`、`英文`、`代码`、`法语`、`西班牙语`等；
5. ✅ 微调后的模型，中文能力显著提升；
6. ✅ 支持不同尺寸`bloom`模型，如`560m`、`3b`、`7b`、`13b`；

## 🥰 体验
1. 🎊 模型已经上传到huggingface上，欢迎来尝试[https://huggingface.co/yuanzhoulvpi/chinese_bloom_560m](https://huggingface.co/yuanzhoulvpi/chinese_bloom_560m)
2. 🎉 在hugginface上部署了一个cpu版本的（有点慢，毕竟是🆓）[https://huggingface.co/spaces/yuanzhoulvpi/chinese_bloom_560_chat](https://huggingface.co/spaces/yuanzhoulvpi/chinese_bloom_560_chat)

## 🔄 模型
1. `bloom`模型支持`中文`、`英文`、`代码`、`法语`、`西班牙语`。具体的训练数据的语言占比如下👇。
![](https://raw.githubusercontent.com/bigscience-workshop/model_card/main/assets/data/pie_chart.svg)

2. `bloom-3b`: [https://huggingface.co/bigscience/bloom-3b](https://huggingface.co/bigscience/bloom-3b)
3. `bloom-系列模型`: [https://huggingface.co/bigscience](https://huggingface.co/bigscience)


## 💽 数据
1. 数据来源于`BelleGroup`，主要是用到这几个数据集:`['BelleGroup/generated_chat_0.4M', 'BelleGroup/school_math_0.25M', 'BelleGroup/train_2M_CN', 'BelleGroup/train_1M_CN',
              'BelleGroup/train_0.5M_CN', 'BelleGroup/multiturn_chat_0.8M']`；
2. 可以基于这些数据样式，制作自己的数据，并训练；




## ⚙️ 步骤

### 数据部分
1. 运行`data_proj/process_data.ipynb`代码；或者模仿结果，制作自己的数据集；
2. 运行结束之后，有一个文件夹`data_proj/opendata`。文件夹下有若干个`json`格式的文件。


### 运行模型
1. 基础运行策略
```python
sh base_run.sh
```
2. `deepspeed`运行策略
```bash
sh ds_all.sh
```
## 🎯 推理代码
1. `infer.ipynb`文件
2. `gradio`交互界面：[https://huggingface.co/spaces/yuanzhoulvpi/chinese_bloom_560_chat](https://huggingface.co/spaces/yuanzhoulvpi/chinese_bloom_560_chat) 因为是使用的huggingface的免费的cpu版本，所以推理速度比较慢。
