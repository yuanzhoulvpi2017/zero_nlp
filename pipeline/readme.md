# pipeline for llama (流水线并行)


### 传统的模型并行👉

![](https://pytorch.org/docs/stable/_images/no_pipe.png)

### 流水线并行👉

![](https://pytorch.org/docs/stable/_images/pipe.png)

## 我的工作

基于pytorch提供的[PIPELINE PARALLELISM](https://pytorch.org/docs/stable/pipeline.html)
功能，参考了transformer包提供的`llama`的代码后，给llama架构实现了一套流水线并行训练代码。

### 整体的gpu负载如下：
![](images/image2.png)

### 动机
1. 之前看到`刘聪nlp`对`chatglm`实现了流水线并行[文章](https://zhuanlan.zhihu.com/p/636488690)，非常羡慕，也想自己写一个。
2. 最近有时间，为了提高自己对`llama`模型结构的认识、提高对transformers代码的了解。
3. 虽然`模型并行`非常好用，但是不能总是用这个吧，太慢了，那就自己实现一个`pipeline`来提高代码能力。
4. 当前，训练框架有很多，比如`deepspeed`、`transformers`、`TencentPretrain`等。也想自己实现一个框架，支持流水线并行、张量并行等。本项目，就当先起个步。目标是自己实现一套支持3d并行的大模型训练库（哈哈哈，单纯口嗨）


### 技术路线
1. 之前，`刘聪nlp`他是用`deepspeed`来实现的，但是我对`deepspeed`不熟悉。因此我在调研了`pippy`、`pytorch`之后，决定，还是用`pytorch`来实现。
2. 其实非常简单，就是把llama的模型代码，从之前的`nn.ModuleList`封装，改为`nn.Sequential`封装。


### 实现的功能
1. 可以直接将`transformers`的`llama`模型转换成拥有流水线并行的`llama`模型。
2. 基础训练流程已经发布（但是比较粗糙）。


### 尚未完成的细节

虽然当前基础的训练流程已经发布，但是比较粗糙，具体体现在：
1. 数据集：我只是提供了随机数生成，但是没做`huggingface`的`datasets`教程，对小白来说，门槛依然有点高。
2. 多gpu负载不合理：当前的网络层分配，负载是不合理的。体现在，最后一张卡的显存占用非常大，还需要优化。
3. 支持`fb16`、`bf16`等精度工作，还没有做。




