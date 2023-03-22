# 📣注意这个文件夹作废，请查看隔壁的📁 `simple_thu_chatglm6b`📣📣




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

| 序号  | 介绍                                       | 文件夹                    | 是否已完成 | 是否还有bug |
|-----|------------------------------------------|------------------------|-------|---------|
| 1   | 使用lora算法对`thuglm-6b`微调                   | `v1_train_thuglm-lora` | ☑️    | ✅       |
| 2   | 使用`transformers`的`Trainer`对`thuglm-6b`微调 | `v2_train_thuglm`      | ☑️    | ✅       |

## 1. 使用lora微调`thuglm-6b`模型 文件夹为`v1_train_thuglm-lora`

<details><summary><b>序号1</b></summary>
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
os.makedirs(name=data_dir, exist_ok=True)

for i in range(20):
    data = pd.DataFrame({'sentence': [
                                         'ChatGLM-6B 是一个开源的、支持中英双语的对话语言模型，基于 [General Language Model (GLM)](https://github.com/THUDM/GLM) 架构，具有 62 亿参数。结合模型量化技术，'] * 100})
    data.to_csv(f"{data_dir}/{i}.csv", index=False)
```

#### 数据注意事项

1. 只要注意，你的数据里面是有一列是文本，这个文本不需要任何标签。比如一列为`sentence`，或者叫`content`。这就可以了。
2. 我们数据加载使用的是`huggingface`的`datasets`包，虽然我们这里使用的是`csv`文件，但是，实际上，你使用`json`格式的数据，都是可以的。
3.
训练大模型，需要的数据肯定也是非常大，担心自己不能处理几百G的数据么？其实不用担心，你只要传递所有的数据的路径即可。剩下的，就可以靠`datasets`
来帮你解决。他会自动对数据做处理，并且对数据所在的位置做内存映射，处理大数据简直是轻飘飘。

这里展示一下加载数据的细节

```python
from glob import glob
from datasets import load_dataset

all_data_list = glob("v1_train_thuglm_lora/data/*")[:10]  # 如果数据大，把这个列表变长一点就行了。

dataset = load_dataset(
    "csv",
    data_files={
        "train": all_data_list[:6],
        "validation": all_data_list[6:],
    },
)
```

### 模型训练

1. `lora`这个算法，已经在`peft`包中实现了。
2. 我看很多人为了使用他，包装了很多代码，实在是看不下去了。这里给一个简单的版本。
3. 这个版本，是模仿`peft`包里面的`examples`的`peft_lora_seq2seq_accelerate_fsdp.py`
   文件写的。因此，在处理tokenizer的部分，可能不太对，但是基本上训练流程已经跑通了。
4. 虽然也是跑通了，但是具体细节上，我还是对`thuglm`
   模型做了修改，主要是为了解决`RuntimeError: expected scalar type Half but found Float`问题。

有些人可能会问，`lora`也没对`thuglm`这类型的模型做支持啊，你这么用，难道不会有问题么？


<details><summary><b>基本上是不会有问题的</b></summary>

1. 查看`lora.py`源码,在`target_modules`里面，有列举了`['q', 'v']`。

```python
# src/peft/tuners/lora.py
@dataclass
class LoraConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.Lora`].

    Args:
        r (`int`): Lora attention dimension
        target_modules (`Union[List[str],str]`): The names of the modules to apply Lora to.
        lora_alpha (`float`): The alpha parameter for Lora scaling.
        lora_dropout (`float`): The dropout probability for Lora layers.
        merge_weights (`bool`):
            Whether to merge the weights of the Lora layers with the base transformer model in `eval` mode.
        fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        enable_lora ( `List[bool]`): Used with `lora.MergedLinear`.
        bias (`str`): Bias type for Lora. Can be 'none', 'all' or 'lora_only'
        modules_to_save (`List[str]`):List of modules apart from LoRA layers to be set as trainable
            and saved in the final checkpoint.
    """

    r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
                    "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
```

2. 查看`transformers`的`T5`模型源码,他里面的`['q', 'v']`对应的是`nn.Linear`层。

```python
# src/transformers/models/t5/modeling_t5.py
class T5Attention(nn.Module):
    # def __init__(self, config: T5Config, has_relative_attention_bias=False):
    #     super().__init__()
    #     self.is_decoder = config.is_decoder
    #     self.has_relative_attention_bias = has_relative_attention_bias
    #     self.relative_attention_num_buckets = config.relative_attention_num_buckets
    #     self.relative_attention_max_distance = config.relative_attention_max_distance
    #     self.d_model = config.d_model
    #     self.key_value_proj_dim = config.d_kv
    #     self.n_heads = config.num_heads
    #     self.dropout = config.dropout_rate
    #     self.inner_dim = self.n_heads * self.key_value_proj_dim

    self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
    self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
    self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
    self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)
```

3. 因此，找到`thuglm`模型中，有关`nn.Linear`层的名称，就可以了。

4. 使用`lora`对`thuglm`模型做修改

```python
from peft import LoraConfig, TaskType, get_peft_model
from peft.utils.other import fsdp_auto_wrap_policy

from train_thuglm.v1_train_thuglm_lora.thuglm.modeling_chatglm import ChatGLMForConditionalGeneration
from train_thuglm.v1_train_thuglm_lora.thuglm.tokenization_chatglm import ChatGLMTokenizer

model = ChatGLMForConditionalGeneration.from_pretrained(
    "THUDM/chatglm-6b", load_in_8bit=False)

tokenizer = ChatGLMTokenizer.from_pretrained("THUDM/chatglm-6b")

# 使用lora模型对thuglm做转换

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1,
    # ['dense','dense_h_to_4h','dense_4h_to_h'] # 'query_key_value',
    target_modules=['dense',
                    'dense_h_to_4h', 'dense_4h_to_h'],
)
model = get_peft_model(model, peft_config)
```

</details>


关键的部分，都已经被列举出来了，剩下的部分，基本上就是和训练`pytorch`模型差不多了，就不再介绍了。

</details>


## 2. 使用`transformers`的`Trainer`对`thuglm-6b`微调
<details><summary><b>序号2</b></summary>

主要做的事情有：
1. 修改了`modeling_chatglm.py`模型源码，可以使用`Tranformers`包的`trainer`来进行训练。
2. 自定义数据。


缺点
1. 需要手动的从huggingface上下载模型依赖的文件到`thu-chatglm-6b`文件夹中，但是要保留我放的`modeling_chatglm.py`文件。
2. 显存消耗大。3090的24G都顶不住。


</details>


