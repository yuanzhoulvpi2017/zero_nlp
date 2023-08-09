# zero to nlp

## 特点

1. 🎯`目标`：基于`pytorch`、`transformers`做中文领域的nlp开箱即用的训练框架，提供全套的训练、微调模型（包括大模型、文本转向量、文本生成、多模态等模型）的解决方案；
2. 💽`数据`：
    - 从开源社区，整理了海量的训练数据，帮助用户可以快速上手；
    - 同时也开放训练数据模版，可以快速处理垂直领域数据；
    - 结合多线程、内存映射等更高效的数据处理方式，即使需要处理`百GB`规模的数据，也是轻而易举；
3. 💻`流程`：每一个项目有完整的模型训练步骤，如：数据清洗、数据处理、模型构建、模型训练、模型部署、模型图解；
4. 🔥`模型`：当前已经支持`gpt2`、`clip`、`gpt-neox`、`dolly`、`llama`、`chatglm-6b`、`VisionEncoderDecoderModel`等多模态大模型；
5. 🚀`多卡串联`
   ：当前，多数的大模型的尺寸已经远远大于单个消费级显卡的显存，需要将多个显卡串联，才能训练大模型、才能部署大模型。因此对部分模型结构进行修改，实现了`训练时`、`推理时`
   的多卡串联功能。
6. ⚙️`模型工具`：添加了大模型的`词表裁切`和`词表扩充`
   教程[model_modify](https://github.com/yuanzhoulvpi2017/zero_nlp/tree/main/model_modify)

## 目录

[//]: # (### 源码解读)

[//]: # ()

[//]: # (当前`transformers`包，确实好用，包括训练等，但是我们不能停留于表面，不能浅尝辄止。要深入源码底部，挖掘出每一个细节。因此，在这个模块中，我将把)

[//]: # (`transfrmers`包中用到的python高级用法、优秀的数据处理思路和方法，尽可能的讲解清楚。)

[//]: # ()

[//]: # (⚠️将逐步完善，敬请期待)

[//]: # (| 模块         | 文件名称 | 作用  | 实现细节 |)

[//]: # (|------------|------|-----|------|)

[//]: # (| Tokenizer  | ☑️   | ☑️  | ☑️   |)

[//]: # (| Datasets   | ☑️   | ☑️  | ☑️   |)

[//]: # (| Model      | ☑️   | ☑️  | ☑️   |)

[//]: # (| Trainer    | ☑️   | ☑️  | ☑️   |)

[//]: # (| AutoClass  | ☑️   | ☑️  | ☑️   |)

[//]: # (| AutoConfig | ☑️   | ☑️  | ☑️   |)

### 模型训练

| 中文名称                              | 文件夹名称                                                                                                                 | 数据 | 数据清洗 | 大模型 | 模型部署 | 图解 |
|-----------------------------------|-----------------------------------------------------------------------------------------------------------------------|----|------|-----|------|----|
| 中文文本分类                            | [chinese_classifier](https://github.com/yuanzhoulvpi2017/zero_nlp/tree/main/chinese_classifier)                       | ✅  | ✅    | ✅   | ❌    | ✅  |
| 中文`gpt2`                          | [chinese_gpt2](https://github.com/yuanzhoulvpi2017/zero_nlp/tree/main/chinese_gpt2)                                   | ✅  | ✅    | ✅   | ✅    | ❌  |
| 中文`clip`                          | [chinese_clip](https://github.com/yuanzhoulvpi2017/zero_nlp/tree/main/chinese_clip)                                   | ✅  | ✅    | ✅   | ❌    | ✅  |
| 图像生成中文文本                          | [VisionEncoderDecoderModel](https://github.com/yuanzhoulvpi2017/zero_nlp/tree/main/vit-gpt2-image-chinese-captioning) | ✅  | ✅    | ✅   | ❌    | ✅  |
| vit核心源码介绍                         | [vit model](https://github.com/yuanzhoulvpi2017/zero_nlp/tree/main/vit)                                               | ❌  | ❌    | ❌   | ❌    | ✅  |
| `Thu-ChatGlm-6b`(`v1`版本 作废)       | [simple_thu_chatglm6b](https://github.com/yuanzhoulvpi2017/zero_nlp/tree/main/simple_thu_chatglm6b)                   | ✅  | ✅    | ✅   | ✅    | ❌  |
| 🌟chatglm-`v2`-6b🎉               | [chatglm_v2_6b_lora](https://github.com/yuanzhoulvpi2017/zero_nlp/tree/main/chatglm_v2_6b_lora)                       | ✅  | ✅    | ✅   | ❌    | ❌  |
| 中文`dolly_v2_3b`                   | [dolly_v2_3b](https://github.com/yuanzhoulvpi2017/zero_nlp/tree/main/chinese_dolly_v2_3b)                             | ✅  | ✅    | ✅   | ❌    | ❌  |
| 中文`llama`(作废)                     | [chinese_llama](https://github.com/yuanzhoulvpi2017/zero_nlp/tree/main/chinese_llama)                                 | ✅  | ✅    | ✅   | ❌    | ❌  |
| 中文`bloom`                         | [chinese_bloom](https://github.com/yuanzhoulvpi2017/zero_nlp/tree/main/chinese_bloom)                                 | ✅  | ✅    | ✅   | ❌    | ❌  |
| 中文`falcon`(注意：falcon模型和bloom结构类似) | [chinese_bloom](https://github.com/yuanzhoulvpi2017/zero_nlp/tree/main/chinese_bloom)                                 | ✅  | ✅    | ✅   | ❌    | ❌  |
| 中文**预训练**代码                       | [model_clm](https://github.com/yuanzhoulvpi2017/zero_nlp/tree/main/model_clm)                                         | ✅  | ✅    | ✅   | ❌    | ❌  |
| 百川大模型                             | [model_baichuan](https://github.com/yuanzhoulvpi2017/zero_nlp/tree/main/model_baichuan)                               | ✅  | ✅    | ✅   | ✅    | ❌  |
| 模型修剪✂️                            | [model_modify](https://github.com/yuanzhoulvpi2017/zero_nlp/tree/main/model_modify)                                   | ✅  | ✅    | ✅   |      |    |
| llama2 流水线并行                      | [pipeline](https://github.com/yuanzhoulvpi2017/zero_nlp/tree/main/pipeline)                                           | ✅  | ✅    | ✅   | ❌    | ❌  |

<details><summary><b>数据流程图解</b></summary>


我一直觉得，数据流程通过图解的形式表达出来，其实是最清楚的，因此我都会尽可能的把每一个任务的都图解出来。

### 文本分类数据图解

![](images/文本分类.003.png)

### 中文gpt2

![](images/chinesegpt2_bot.png)

### 中文clip

![model](images/clip001.png)

### 图像生成中文文本

![model](images/vision-encoder-decoder.png)

### vit 源码

![](images/vit_architecture.jpg)
</details>

# 分享数据

一直在整理开源数据，如果有需要，可以关注公众号`统计学人`，回复`nlp数据`即可。目前还在整理数据中

![统计学人](images/gzh.jpg)