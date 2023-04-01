# 单机多卡、`模型并行`方式训练`thu/chatglm6b`模型



## 介绍
1. 本文件夹在`v1`[simple_thu_chatglm6b](https://github.com/yuanzhoulvpi2017/zero_nlp/tree/main/simple_thu_chatglm6b)的基础上，添加了单机多卡的训练代码 ：
2. **模型并行（将大模型的各个层分别放在多个显卡上）**
3. 同时，结合`lora`算法、`fp16`精度、使用`checkpoint`等方法，可以在文本长度为`1024`、`batchsize=4`的情况下，在两个T4显卡上跑的很快乐（显卡的显存最大为16G，但是实际上卡1用了8G，卡2用了11G），甚至batchsize还可以提高。
4. 虽然`thu/chatglm6b`也给了基于`p-tuning`的微调代码，但是和我这个单机多卡比起来，dddd（懂得都懂），各取所需吧。

# 准备步骤和详细的说明


## 模型文件准备
<details><summary><b>和v1一样</b></summary>

为了有条理性，我把这个模型的所有代码全部都放在📁`thuglm`文件夹下。
![](images/截屏2023-03-22%2019.08.54.png)

但是，你在从github上下载我这个仓库后，是看不到这几个文件的：
1. `pytorch_model-00001-of-00008.bin`、
2. `pytorch_model-00002-of-00008.bin`、
3. `pytorch_model-00003-of-00008.bin`、
4. `pytorch_model-00004-of-00008.bin`、
5. `pytorch_model-00005-of-00008.bin`、
6. `pytorch_model-00006-of-00008.bin`、
7. `pytorch_model-00007-of-00008.bin`、
8. `pytorch_model-00008-of-00008.bin`、
9. `ice_text.model`

你需要从[https://huggingface.co/THUDM/chatglm-6b/tree/main](https://huggingface.co/THUDM/chatglm-6b/tree/main) 这里把上面列举的文件下载下来。

注意查看，在这个链接里面，每个文件后面都有一个下载的箭头
![](images/截屏2023-03-22%2019.06.22.png)


**下载后，把下载的文件都放在`thuglm`文件夹下面，然后和我的截图比对一下，是不是有什么出入。**

到这里，模型部分就解决了。
</details>

## 数据部分
1. 很多人都喜欢使用那个羊驼数据，无所谓，反正我这里都已经支持了。可以使用这个人提供的语料
```bash 
git clone https://github.com/hikariming/alpaca_chinese_dataset.git 
```

## 训练代码`train_model_all.py`
这个代码主要分下面几个部分
1. `数据准备`：主要是对数据做转换。和`v1`一样的
2. `模型加载和转换`：注意这里有个`device_map_dict`字典，你需要手动映射`cuda`序号。
3. `数据加载`：没啥好说的
4. `训练`

### 注意
1. 其实这个代码和`v1`版本的代码差不多，就是在`模型加载和转换`部分，和`v1`不一样。
2. 不知道是`modeling_chatglm.py`代码写的有问题，还是在加载模型的时候出现`bug`，反正就是很奇怪，只能让你手动分配。
3. 另外，我基于`pytorch`的`模型并行`思路，还对`modeling_chatglm.py`里面的模型的`forward`方法，做了调整，让每个网络层的`input`数据自动切换到该网络层所在的设备上。


### 多个卡怎么调整
1. 调整`模型加载和转换`的`device_map_dict`字典的值，值的顺序从低到高升序。
2. 因为我只用了两个卡，所以映射的`cuda`序号只是为`0`、`1`，如果你设备更多，你自己添加。
3. 同时也别忘记修改代码前面的`os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"`部分了。
4. 因为最后一个卡，还需要计算`loss`部分，所以，在分配的时候，最后一个卡最好不要分配太多的网络层。
5. 建议多试一试。找到一个最好的参数。

## 推理部分
1. 推理部分，和`v1`版本一样，这个不再细聊
2. 关于模型效果等，这个也去看`v1`，基本上也都没问题了。


