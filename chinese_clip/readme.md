# 介绍

1. 本文将介绍，如何从0到1的训练一个中文clip模型。
2. 在处理数据的过程中，训练的过程中，需要的注意事项。
3. 从数据流的角度，看看clip模型是怎么处理数据的，模型是怎么构建的。image和text的模型的差异性，两个模型是怎么合并起来计算loss的。

# clip模型介绍

CLIP的英文全称是Contrastive Language-Image Pre-training，即一种基于对比文本-图像对的预训练方法或者模型。
CLIP是一种基于对比学习的多模态模型，与CV中的一些对比学习方法如moco和simclr不同的是，
CLIP的训练数据是`文本-图像对`：一张图像和它对应的文本描述，这里希望通过对比学习，
模型能够学习到文本-图像对的匹配关系。
如下图所示，CLIP包括两个模型：

1. Text Encoder和Image Encoder，其中Text Encoder用来提取文本的特征，可以采用NLP中常用的text transformer模型；
2. Image Encoder用来提取图像的特征，可以采用常用CNN模型或者vision transformer。
   ![model](images/clip001.png)

上面这段文字来源于[https://zhuanlan.zhihu.com/p/493489688](https://zhuanlan.zhihu.com/p/493489688)

### 大白话

1. 从数据上看：之前相似度计算，都是两个文本对：`text - text`。只不过现在都是`text - image`了。
2. clip是两个模型（具体长什么样子，后面再说）

- 2.1 `text-model`：负责把`text`转换成向量。
- 2.2 `image-model`：负责把`image`转换成向量。
- 2.3 然后把上面两个向量，做交叉计算loss，然后loss反向传播，这样两个模型的参数都会更新。

3. 其实你想啊，这个`image-model`处理图像的，其实也可以改为处理视频、处理3d模型等。那简直是格局打开🫴了。我现在没有数据，后面也打算做一个。
4. 你再想想，`text-image` => `text-image-video-3d`这样联合起来，是不是更好。没数据，没机器，做不了。
5. 有些人可能感觉，`你这人，就知道TMD吹牛`，来来来，我带你研究研究clip模型的源码。

# `transfromers`包的`clip`源码

计算机这行业就是这样，你文字写的天花乱坠，都不如直接去看看源码。因为看源码，可以帮你了解，这些数学算法到底是怎么实现的。

【待完善】

# 工程方面

## 数据

1. 数据来源于公众号`YeungNLP`，关注他，并且回复`005`
   即可获得。当然也可以直接点击链接[https://pan.baidu.com/s/1wGmXUNP021OWnW7Kik7q1A?pwd=gd3c
   ](https://pan.baidu.com/s/1wGmXUNP021OWnW7Kik7q1A?pwd=gd3c)来获得。
2. 把下载好的文件，也就是`test-2.6w.csv`、`train-137w.csv`放在文件夹📁`bigdata/raw_data`里面。
3. 以此运行`processdta_01.ipynb`、`processdta_02.ipynb`、`processdta_02.ipynb`用来处理数据。

- 3.1 `processdta_01.ipynb`：用来下载数据，大概下载了10多个小时。
- 3.2 `processdta_02.ipynb`：用来筛选数据，不是所有的图片数据都是可以用的，这一步非常坑。需要留意。如果图片没有筛选好，在你训练到中间的时候，突然一下因为图片无法加载导致错误，从而训练中断了。
- 3.3 `processdta_03.ipynb`：用来把数据干净的数据处理好，合并好，生成新的，漂亮的训练数据。

4. 其实完整下来看，数据清洗，就是把符合格式的照片筛选出来，然后进行训练。









