# 介绍
本部分，介绍中文的文本分类模型，适用于二分类、多分类等情况。使用transformers库。

## 1. 处理数据`code_01_processdata.ipynb`
### 数据介绍
1. 本案例使用的是一个外卖平台的评论数据，对评论的文本做了分类（分为好评和差评）
2. 当你把`code_01_processdata.ipynb`文件跑完之后，就可以看到在📁`data_all`里面有一个📁`data`，里面有三个文件，样式都是像下面👇这样的

<img src="https://github.com/yuanzhoulvpi2017/zero_nlp/raw/main/images/%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB.002.png"/>

上图是一个`batch`的数据，或者所有的文本分类的数据样式：
1. `text`下面的红色条，就是一个个句子。
2. `label`里面有红色有绿色，就是表示标签分类。
3. `transformers`包做分类的时候，数据要求就这两列。


注意点：
1. 数据需要分为`train_data.csv`,`test_data.csv`,`valid_data.csv`,这三个`csv`文件注意是使用`,`分割开的。
2. 数据不可以有缺失值
3. 数据最好只含有两列：`label`,`text`
 - `label`:表示标签，最好为整型数值。0,1,2,3,4等
 - `text`:表示文本，（看你需求，可以有符号，也可以没有标点符号）
4. `train_data.csv`,`test_data.csv`,`valid_data.csv`这三个数据里面，不要有数据相同的，不然会造成数据泄漏。


## 2. 训练模型`code_02_trainmodel.ipynb`
<img src="https://github.com/yuanzhoulvpi2017/zero_nlp/raw/main/images/%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB.003.png"/>

### 数据训练流程
以一个batch为例：
1. `Tokenizer`会将数据中的`text`转换成三个矩阵(或者叫三个`Tensor`)，分别叫`input_ids`,`token_type_ids`,`attention_mask`，至于怎么转换的，我们先不做详细介绍（本仓库后续会介绍）。
2. `pretrained model`在被加载之前，需要设置一大堆模型的参数，至于要设置什么参数，我们也不做详细介绍。
3. `Trainer`就是一个训练器，也需要预先设置好一大堆参数。至于要设置什么参数，我们也不做详细介绍。
4. `Trainer`会把`input_ids`,`token_type_ids`,`attention_mask`；还有数据自带的标签`label`；还有`pretrained model`都加载进来，进行训练；
5. 当所有batch的数据更新完之后，最终就会生成一个模型。`your new model`就诞生了。
6. 对于刚开始学习`大模型做nlp分类`的任务，其实不需要考虑那么多细节，只需要注意数据流程。

注意点：
1. 这个步骤非常看显存大小。显卡显存越大越好。`batch_size`,`eval_size`大小取决于显存大小。
2. 在实际工程中，会先使用`Tokenizer`把所有的文本转换成`input_ids`,`token_type_ids`,`attention_mask`，然后在训练的时候，这步就不再做了，目的是减少训练过程中cpu处理数据的时间，不给显卡休息时间。
3. 在使用`Tokenizer`把所有的文本做转换的期间，如果设置的文本的长度上限为64，那么会把大于64的文本截断；那些少于64的文本，会在训练的时候，在喂入模型之前，把长度补齐，这么做就是为了减少数据对内存的占用。



## 3. 预测`code_03_predict.ipynb`

1. 这个时候，就是搞个句子，然后丢给一个`pipeline`(这个就是把`Tokenizer`和`你的大模型`放在一起了)，然后这个`pipeline`就给你返回一个分类结果。
2. 常见的就是使用`pipeline`，如果更加复杂的话，比如修改模型，这个时候，就比较复杂了（后面会再次介绍）。

## 4. 部署

1. 简单的`部署`相对于`预测`，其实就是再加一层web端口，fastapi包就可以实现。
2. 高级一点的`部署`相对于`预测`，就需要把模型从`pytorch`转换成`onnx`格式的，这样可以提高推理效率（也不一定，就是举个例子），可能也不会使用web端口（http协议）了，会使用rpc协议等方法。这部分现在先不看。
