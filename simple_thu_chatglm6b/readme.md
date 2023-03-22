# 🚀 最简单的方法训练`thuglm-6b`模型 🎯

1. 💻一个3090消费级的显卡就可以训练
2. 🎯支持`tensorboard`等各种花里胡哨小插件
3. 🚀也可以多卡并行，训练非常快
4. ✅数据只需要文本即可，不管是json还是csv文件，都可以，无监督学习，整理数据更轻松
5. 📝训练代码比以往的教程更加简单，可以说是最简单的训练`thuglm-6b`教程了


## 我做了什么，有什么效果
只是对`transofrmers`包的`Trainer`类做了修改，对`modeling_chatglm.py`代码也做了修改。
这么做，可以让你在拥有22G显存的情况下，可以训练`thu-chatglm-6b`模型。

那么，基于`Trainer`的丰富方法，你可以做很多事情。而且使用`peft`包的`lora`算法，让你在一个消费级别的显卡上，就可以训练`thu-chatglm-6b`模型。

# 教程

## 模型部分

为了有条理性，我把这个模型的所有代码全部都放在📁`thuglm`文件夹下。
![](images/截屏2023-03-22%2019.08.54.png)


但是，你在从github上下载我这个仓库后，是看不到这几个文件的：
1. `pytorch_model-00001-of-00008.bin`、
2. `pytorch_model-00002-of-00008.bin`、
3. `pytorch_model-00002-of-00008.bin`、
4. `pytorch_model-00003-of-00008.bin`、
5. `pytorch_model-00004-of-00008.bin`、
6. `pytorch_model-00005-of-00008.bin`、
7. `pytorch_model-00006-of-00008.bin`、
8. `pytorch_model-00007-of-00008.bin`、
9. `pytorch_model-00008-of-00008.bin`、
10. `ice_text.model`

你需要从[https://huggingface.co/THUDM/chatglm-6b/tree/main](https://huggingface.co/THUDM/chatglm-6b/tree/main) 这里把上面列举的文件下载下来。

注意查看，在这个链接里面，每个文件后面都有一个下载的箭头
![](images/截屏2023-03-22%2019.06.22.png)


**下载后，把下载的文件都放在`thuglm`文件夹下面，然后和我的截图比对一下，是不是有什么出入。**

到这里，模型部分就解决了。
## 数据部分

我这里给一个样本数据，就是单纯参考：

**链接：https://pan.baidu.com/s/1HZoEofUmXgq68-1sqZNVTw?pwd=1u20 
提取码：1u20**

里面有一个名叫`data2.zip`的压缩包文件，直接解压到当前文件夹就行了。

`data2`展开是这样的：

![](images/截屏2023-03-22%2019.17.13.png)

`data2`在整个文件系统上来看，是这样的：

![](images/截屏2023-03-22%2019.18.07.png)


### 数据详解
1. 注意到数据里面是有一列，叫`content`
2. 你想换成别的数据都是可以的，本质上是使用的`datasets`这个包，也是`huggingface`出品的。


# 安装

上面是文件工程，这里开始说安装包，直接使用`pip`安装

```bash
pip install protobuf==3.20.0 transformers icetk cpm_kernels peft
```

就这么简单，不需要安装别的东西了

# 训练部分
训练部分，直接运行`train_chatglm6b.py`代码，就可以了，但是这里，直接在写一次详细的讲解。

# 推理部分
1. 推理部分，直接看`infer.ipynb`代码


