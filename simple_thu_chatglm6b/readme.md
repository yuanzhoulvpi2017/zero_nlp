# 🚀 最简单、最便宜的训练`thu-chatglm-6b`模型教程 🎯


# 📝 更新记录
## **03-28 版本**
1. 解决了`03-27`版本中、在部分设备上、进行单机多卡计算的时候，出现的`TypeError: 'NoneType' object is not subscriptable`问题
2. 解决了`03-24`版本中、训练了，但是没效果的问题
3. 添加了一整套的完整的训练代码`code02_训练模型全部流程.ipynb`,使用alpaca数据集格式.
4. 感谢`https://github.com/hikariming/alpaca_chinese_dataset`整理的数据

## **03-27 版本**
1. 🚀**添加了多卡并行的功能**
2. ✅会基于你的显卡数量，自动进行并行计算，也可以自己选择哪些卡，在代码的`train_chatglm6b.py`文件的前两行代码
3. 😘我做的事情：就是改了我就是修改了`thuglm/modeling_chatglm.py`代码，对里面涉及到的变量，做了设备的指定（虽然原始的代码也做了，但是做了并不充分）
4. 🤗本质上，使用的就是pytorch的`nn.DataParallel`功能,因为我就是想让他支持`transformers`的`Trainer`。

### ⛔️注意事项
1. 在使用的时候，第一张卡的压力要大一点。
2. 我在测试的时候，发现在3个3090上，是完全没有问题的。但是在4个3090的时候，会出现小bug：`RuntimeError: CUDA error: an illegal memory access was encountered`（说明我的device分配依然不太对）。
3. ~~我在两个T4的机器上训练，会出现一个小bug:`TypeError: 'NoneType' object is not subscriptable`（这个应该是我的代码不对）~~
4. 虽然bug不少，但是可以知道在什么地方优化，知道改哪里了，后面将继续优化！！！🎯 冲！！！
5. 各位大佬，多提一提bug，让小弟来改。

## **03-24 版本**
1. 💻 现在可以在16G显存的显卡上进行训练（在`batchsize=1,content_length=512`的情况下）
2. 🚀使用了`torch.utils.checkpoint`，降低了显存的占用（从之前的24G降低到15.2G左右），但是训练的时间花费更多。（如果你想关闭这个功能，在`thuglm/modeling_chatglm.py`文件的第`713`行`self.gradient_checkpointing = True`中，把`True`改为`False`即可）
3. 🤖 精度依然是使用的`fp16`，而不是`int8`.
4. 💨 依然使用了`lora`方法，如果不想使用这个方法，我后续可以把这个方法关闭。
5. 📣 现在你可以把`content_length`调整到`1024`， `batchsize`可以调整到`4`，即使这样，显存依然维持在23G左右。
![](images/WechatIMG15931.jpeg)

## **03-22 版本**
1. 💻一个3090消费级的显卡就可以训练
2. 🎯支持`tensorboard`等各种花里胡哨小插件
3. 🚀也可以多卡并行，训练非常快
4. ✅数据只需要文本即可，不管是json还是csv文件，都可以，无监督学习，整理数据更轻松
5. 📝训练代码比以往的教程更加简单，可以说是最简单的训练`thu-chatglm-6b`教程了


## 我做了什么，有什么效果
只是对`transofrmers`包的`Trainer`类做了修改，对`modeling_chatglm.py`代码也做了修改。
这么做，可以让你在拥有22G显存的情况下，可以训练`thu-chatglm-6b`模型。

那么，基于`Trainer`的丰富方法，你可以做很多事情。而且使用`peft`包[https://github.com/huggingface/peft](https://github.com/huggingface/peft)的`lora`算法，让你在一个消费级别的显卡上，就可以训练`thu-chatglm-6b`模型。

# 教程

## 模型部分

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

# ✅ 训练部分

## 🎯 **在最新的版本中，只需要查看`code02_训练模型全部流程.ipynb`文件就行了**


# ✅ 推理部分
1. 推理部分，直接看`infer.ipynb`代码
2. 能到这里，也是恭喜你，微调模型已经成功了。这个时候，在这个文件夹下，肯定有一个文件夹叫`test003`（就是上面`output_dir="test003"`对应的文件夹）
3. 在这个文件夹下，你肯定可以看到很多`checkpoint-xxx`，选择一个你喜欢的（当然，肯定是最好选择最新的）。



# 🎯
1. 你只需要拥有一个3090即可（只要显存有24G就行了）
2. 目前还没有尝试过多卡，下次去公司试一试

