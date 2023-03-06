# 前言
1. 之前都搞过`clip`、`image-encoder-decoder`。现在哪里还怕搞不懂`vit`.
2. 这里主要分享一下`vit`的最核心的部分。

## vit 核心的数据内容

vit想法非常牛，但是数据处理的思想更牛，之前都没提出来过。

![](images/vit_architecture.jpg)

载对于一个图片，将一个图片分割成N块。巧妙的使用`nn.Conv2d`。


### 初始化
```python
import torch
from torch import nn 

# base parameter

image_size=224 # 图片的width和height
patch_size=16  # 将图片的分为块，每一块的大小为16x16，这样就有(224//16)^2 = 14 ^2 = 196个
num_channels=3 #  R,G, B
hidden_size=768 # 输出的hidden_size
batch_size = 16 # 一批数据有多少
```

### 创建一个分块器和一个样本数据(一个`batch`)
```python
# 分块器
project = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

# 样本数据(一个`batch`) 
# batch_size, num_channels, height, width = pixel_values.shape
pixel_values = torch.randn(batch_size, num_channels, image_size, image_size)

pixel_values.shape 
```

### 输出分块的大小
```python
project(pixel_values).shape 

#> torch.Size([16, 768, 14, 14])
```

### 数据再转换一下，image的embedding就完成了。
```python
image_embedding = project(pixel_values).flatten(2).transpose(1, 2)
image_embedding.shape 
#> torch.Size([16, 196, 768]) # batch_size, seq_length, embedding_dim
```

这个时候，就已经和文本的数据一样了。维度都是(`batch_size, seq_length, embedding_dim`)，再向下推导，就是`transformers`了。没什么可介绍的了。



