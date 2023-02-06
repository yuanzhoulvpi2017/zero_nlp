# 介绍
本部分，介绍中文的文本分类模型，适用于二分类、多分类等情况。使用transformers库。

## 1. 处理数据`code_01_processdata.ipynb`
<img src="https://github.com/yuanzhoulvpi2017/zero_nlp/raw/main/images/%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB.002.png"/>

注意点：
1. 数据需要分为`train_data.csv`,`test_data.csv`,`valid_data.csv`
2. 数据不可以有缺失值
3. 数据最好只含有两列：`label`,`text`
 - `label`:表示标签，最好为整型数值。0,1,2,3,4等
 - `text`:表示文本，（看你需求，可以有符号，也可以没有标点符号）
4. `train_data.csv`,`test_data.csv`,`valid_data.csv`这三个数据里面，不要有数据相同的，不然会造成数据泄漏。


## 2. 训练模型`code_02_trainmodel.ipynb`
<img src="https://github.com/yuanzhoulvpi2017/zero_nlp/raw/main/images/%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB.003.png"/>

### 数据训练流程
以一个batch为例：
1. `Tokenizer`会将数据中的`text`转换成三个矩阵(或者叫三个`Tensor`)，分别叫`input_ids`,`token_type_ids`,`attention_mask`
2. `pretrained model`在被加载之前，需要设置一大堆模型的参数。
3. `Trainer`就是一个训练器，也需要预先设置好一大堆参数。
4. `Trainer`会把`input_ids`,`token_type_ids`,`attention_mask`；还有数据本来自带的标签`label`；还有`pretrained model`都加载进来，进行训练；
5. 当所有batch的数据更新完之后，最终就会生成一个模型。`your new model`就诞生了。


注意点：
1. 这个步骤非常看显存大小。显卡显存越大越好。
2. `batch_size`,`eval_size`大小取决于显存大小。

## 3. 预测`code_03_predict.ipynb`

## 4. 部署