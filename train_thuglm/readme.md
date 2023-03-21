# 训练`thuglm-6b`模型

# `thuglm-6b`模型和`gpt2`模型的差异

## loss部分

1. 查看了`thuglm-6b`模型源码，他的loss和`gpt2`等自回归模型的loss，基本上是一样的。(这里只是考虑自回归类型的训练)

```python
# 
# 这是thuglm模型的loss
if labels is not None:
    lm_logits = lm_logits.to(torch.float32)

    # Shift so that tokens < n predict n
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    lm_logits = lm_logits.to(hidden_states.dtype)
    loss = loss.to(hidden_states.dtype)
```

```python
# src/transformers/models/gpt2/modeling_gpt2.py 的class GPT2LMHeadModel(GPT2PreTrainedModel):
# 这是gpt2的loss 
loss = None
if labels is not None:
    # Shift so that tokens < n predict n
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


```

## 代码风格

1. `thuglm-6b`源码和`transformers`包的`gpt2`源码，长得非常像，设计模式是一摸一样的。从工程角度来看，你只要看过`gpt2`
   的源码，看懂了，那么`thuglm-6b`的代码框架对你来说肯定不难。
2. 数学角度来说，这个我没有看过两个模型的论文，不敢胡说，这部分我就不解释了。

## 数据角度

1. `thuglm-6b`模型和`transformers`包的`gpt2`源码里面的模型，在`forward`方法里面，需要的参数，基本上是保持一致的，因此。需要的数据样式，也都差不多。
2. 那么虽然现在`thuglm-6b`还没有所谓的`thuglmForSequenceClassification`、`thuglmForTokenClassification`
   等方法，但是直接模仿`gpt2`的风格来写，就行了。就是`loss`更改一下，下游层更改一下。

## 本人对`thuglm-6b`模型的态度

1. `thuglm-6b`
   模型，最近太火了，而且在中文语言的表现上，效果非常好[https://github.com/THUDM/ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B)
   ，使用int8还可以在小显存上进行推理，非常amazing。
2. 目前，很难在在市面上找到非常好的中文`gpt2`模型，可能是数据方面的问题，或者机器方面的问题。
3. 在我眼里，我其实就是把他当成一个在中文领域表现非常好的`gpt2`模型而已。（抛开别的条件不谈）。

# 训练`thuglm-6b`模型

| 序号  | 介绍                     | 文件夹                    | 是否已完成 | 是否还有bug |
|-----|------------------------|------------------------|-------|---------|
| 1   | 使用lora算法对`thuglm-6b`微调 | `v1_train_thuglm-lora` | ☑️    | ✅       |
| 2   | 单卡直接对`thuglm-6b`微调     | `v2_train_thuglm`      | ☑️    | ✅       |
| 3   | 多卡对`thuglm-6b`微调       | `v3_train_thuglm`      | ☑️    | ✅       |

## 1. 使用lora微调`thuglm-6b`模型 文件夹为`v1_train_thuglm-lora`

1.目前，训练一个`thuglm-6b`模型，还是比较费劲的（我还没试过，目前都在传使用lora方法来进行训练）。那也就跟风写一个教程。

2. 文本，将介绍如何使用`peft`[https://github.com/huggingface/peft](https://github.com/huggingface/peft)
   包（这个包实现了`lora`算法）、对`thuglm-6b`进行微调。
3. 硬件设备是3090（显存为24G）。
4. 包括数据整理、模型转换、训练加载等详细步骤。

### 数据部分
在前面也说到，`thuglm-6b`的`ChatGLMForConditionalGeneration`loss和`gpt2`的`GPT2LMHeadModel`loss是差不多的，都是自回归模型，就是名字不一样而已。

因此，可以看看我的`chinese-gpt2`模型训练的数据要求。

<details><summary><b>chinese-gpt2模型数据</b></summary>

#### 数据来源

1. 获得数据:数据链接，关注公众号【`统计学人`】，然后回复【`gpt2`】即可获得。

#### 数据格式

1. 数据其实就是一系列文件夹📁，然后每一个文件夹里面有大量的文件，每一个文件都是`.csv`格式的文件。其中有一列数据是`content`
2. 每一行的`content`就代表一句话,截图如下
   <img src="https://github.com/yuanzhoulvpi2017/zero_nlp/raw/main/images/chinesegpt2_data.png"/>
3. 虽然数据有15GB那么大，但是处理起来一点也不复杂，使用 `datasets`
   包，可以很轻松的处理大数据，而我只需要传递所有的文件路径即可，这个使用 `glob` 包就能完成。
</details>


当然，也可以直接生成一个数据，可以这么写

```python
import numpy as np
import pandas as pd
import os

data_dir = "data"
os.makedirs(name=data_dir,exist_ok=True)

for i in range(20):
    data = pd.DataFrame({'sentence':['ChatGLM-6B 是一个开源的、支持中英双语的对话语言模型，基于 [General Language Model (GLM)](https://github.com/THUDM/GLM) 架构，具有 62 亿参数。结合模型量化技术，'] * 100})
    data.to_csv(f"{data_dir}/{i}.csv", index=False)
```

### 数据注意事项
1. 只要注意，你的数据里面是有一列是文本，这个文本不需要任何标签。比如一列为`sentence`，或者叫`content`。这就可以了。
2. 我们数据加载使用的是`huggingface`的`datasets`包，虽然我们这里使用的是`csv`文件，但是，实际上，你使用`json`格式的数据，都是可以的。
3. 训练大模型，需要的数据肯定也是非常大，担心自己不能处理几百G的数据么？其实不用担心，你只要传递所有的数据的路径即可。剩下的，就可以靠`datasets`来帮你解决。他会自动对数据做处理，并且对数据所在的位置做内存映射，处理大数据简直是轻飘飘。


这里展示一下加载数据的细节
```python
from glob import glob
from datasets import load_dataset
all_data_list = glob("v1_train_thuglm_lora/data/*")[:10]

dataset = load_dataset(
    "csv",
    data_files={
        "train": all_data_list[:6],
        "validation": all_data_list[6:],
    },
)
```










