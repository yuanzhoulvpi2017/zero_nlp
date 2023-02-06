# 介绍
本部分，介绍中文的文本分类模型，适用于二分类、多分类等情况。使用transformers库。

## 1. 处理数据`code_01_processdata.ipynb`
![数据样式](images/文本分类.002.png)

注意点：
1. 数据需要分为`train_data.csv`,`test_data.csv`,`valid_data.csv`
2. 数据不可以有缺失值
3. 数据最好只含有两列：`label`,`text`
 - `label`:表示标签，最好为整型数值。0,1,2,3,4等
 - `text`:表示文本，（看你需求，可以有符号，也可以没有标点符号）
4. `train_data.csv`,`test_data.csv`,`valid_data.csv`这三个数据里面，不要有数据相同的，不然会造成数据泄漏。


## 2. 训练模型`code_02_trainmodel.ipynb`
注意点：
1. 这个步骤非常看显存大小。越大越好。
2. `batch_size`,`eval_size`大小取决于显存大小。

## 3. 预测`code_03_predict.ipynb`

## 4. 部署