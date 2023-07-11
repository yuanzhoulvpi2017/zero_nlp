# 模型修改工具集合

## 词表扩充

1. 场景：
- 1.1 对国外的大模型，添加新的中文词表；
- 1.2 对通用大模型，添加垂直领域的词表；
2. 实现逻辑：
- 2.1 准备特定领域的文本进行分词，得到一串词表列表（不在乎形式，最后数据形式为`List[str]`即可，比如：`['疎', '很多人都', '樣的', '商人', '藥物']`)
- 2.2 使用`tokenizer.add_tokens`方法，将上述词表添加上去。
- 2.3 使用`model.resize_token_embeddings`方法，重新调整`embedding`的大小。
- 2.4 将`tokenizer`和`model`保存起来。
- 2.5 冻结部分参数，只训练`embedding`对应的参数。

前四步，参考代码:[`code01_扩充词表.ipynb`](https://github.com/yuanzhoulvpi2017/zero_nlp/blob/main/model_modify/code01_%E6%89%A9%E5%85%85%E8%AF%8D%E8%A1%A8.ipynb)

## 词表裁切

1. 场景：
- 1.1 对多语言模型，做词表裁切（丢掉用不到的词表），比如只保留bloom模型的中文部分；

2. 实现逻辑：
- 2.1 查看模型的所有词表；
- 2.2 找到不用的词表；
- 2.3 把`tokenizer`里面的词表删掉，并且重新建立索引；
- 2.4 把`model`里面的`embedding`和`lm_head`的矩阵的行或者列删掉；
- 2.5 保存

参考代码:[https://github.com/yuanzhoulvpi2017/LLMPruner](https://github.com/yuanzhoulvpi2017/LLMPruner)
