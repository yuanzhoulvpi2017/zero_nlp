# 自定义算子实现跨卡通讯让batch更大！loss视野更广！ 基于ddp使用对比学习训练图文embedding模型（chinese-clip）

1. 关联的b站视频：[https://www.bilibili.com/video/BV1xTz1YVEmV](https://www.bilibili.com/video/BV1xTz1YVEmV) 建议结合B站视频一起使用！！！


## 跨卡通讯 让batch更大
1. 直接查看代码：https://github.com/yuanzhoulvpi2017/zero_nlp/blob/main/chinese_clip_ddp/embedding/model.py#L286



## 基于vit做模型的二次开发
1. 直接查看代码：https://github.com/yuanzhoulvpi2017/zero_nlp/blob/main/chinese_clip_ddp/embedding/vit4embedding.py



## 不同的跨卡通讯方式，效果不一样的。有的是报错、有的是loss不下降

1. 使用小数据训练，loss对比
![image](loss_vs_step_by_model_type.png)


2. 使用更大的数据训练，loss对比
![image](loss_vs_step_by_model_type_big.png)