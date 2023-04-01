
# 本文件夹实现了：
1. **模型并行（将大模型的各个层分别放在多个显卡上）**
2. lora(基于peft)
3. fp16
4. 文本长度为`1024`、`batchsize=4`的情况下，在两个T4显卡上跑的很快乐（显卡的显存最大为16G，但是实际上卡1用了8G，卡2用了11G），batchsize还可以提高。
5. 使用了`checkpoint`降低显存。



## 文件准备

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

