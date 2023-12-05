## 结合lora 对百川2-7b 做dpo
这个脚本是基于`trl`包的一个`example`修改的,原版的链接为:[train_dpo.py](https://github.com/huggingface/trl/blob/cbc6c9bb3ebe810efeb34883806169ed7ce338a4/examples/scripts/dpo.py)


### 主要是做了下面几点改动：
1. 使用的数据是经过处理的,因为没有办法直接下载`hh/rlhf`数据，而且刚开始也是为了研究这个数据的样式是什么样子的。
   - 另外，因为`hh/rlhf`数据的prompt形式是`\n\nHuman: `、`\n\nAssistant: `，`baichuan2-chat`模型的prompt是`<reserved_106>`、`<reserved_107>`，所以需要做一部分转换。
   - 关于如何自定义自己的数据，后面会出详细教程。
2. 使用的模型是`baichuan2-7b-chat`
3. 训练的框架使用的是`trl`包，这个是huggingface开发的，和`transformers`是一脉相承。
   - 现在训练大模型，支持最好的框架就是`transformers`。那么，基于这个框架做的二次开发的包，上手就简单很多。
   - 这个包在强化学习里面，确实也是最流行的。
4. 训练的时候，是使用`lora`来训练，因为`trl`的`dpoTrianer`是做了优化的。
   - 当`model` 是`peftmodel`类型的时候（也就是加了一层`lora`)，且`model_ref`是None的时候，会`model_ref`默认等于`model.disable_adapter()`（也就是把模型套的那层lora给扒掉）。
   

## 使用教程

### 数据部分

#### 1. 直接使用官方提供的demo数据

```shell
bash data01_download_hhrlhf.py

```

#### 2. 使用自定义数据

待更新

### 训练模型

```shell
sh train_ds.sh

```



### `QA`
`Q`：为什么使用`baichuan2`模型呢？

`A`：因为`baichuan2`模型，在同等参数量的情况下，效果最好。