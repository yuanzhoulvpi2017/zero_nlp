# 🚀 最简单、最便宜的训练`thu-chatglm-6b`模型教程 🎯 
1. 感谢智谱AI开源`chatglm-v2-6b`大模型；
2. 之前就给`v1`版本做过lora，在智谱AI宣布`v2`可以商用后，打算给`v2`也做一版lora；
3. 基于`v2`的[官网代码](https://github.com/THUDM/ChatGLM2-6B/tree/main/ptuning)，做了简单修改；

## 📝 更新记录
1. **07-14 版本** `chatglm-v2-6b`模型的`lora`训练方案🔗👉[**chatglm_v2_6b_lora**](https://github.com/yuanzhoulvpi2017/zero_nlp/tree/main/chatglm_v2_6b_lora)

# 🔄 训练

## 下载数据集
ADGEN 数据集任务为根据输入（content）生成一段广告词（summary）。

```json
{
    "content": "类型#上衣*版型#宽松*版型#显瘦*图案#线条*衣样式#衬衫*衣袖型#泡泡袖*衣款式#抽绳",
    "summary": "这件衬衫的款式非常的宽松，利落的线条可以很好的隐藏身材上的小缺点，穿在身上有着很好的显瘦效果。领口装饰了一个可爱的抽绳，漂亮的绳结展现出了十足的个性，配合时尚的泡泡袖型，尽显女性甜美可爱的气息。"
}
```

从 [Google Drive](https://drive.google.com/file/d/13_vf0xRTQsyneRKdD1bZIr93vBGOczrk/view?usp=sharing) 或者 [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/b3f119a008264b1cabd1/?dl=1) 下载处理好的 ADGEN 数据集，将解压后的 `AdvertiseGen` 目录放到本目录下。

## 硬件要求
1. **有个`3090`显卡即可（24G显存左右）**
2. 在下面这个参数下，显存只需要`14G`
```sh
    --max_source_length 64 \
    --max_target_length 128 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \ 
    --lora_r 32

```

## 训练脚本

1. 使用vscode调试，就在`.vscode/launch.json`里面；
2. 直接使用sh，`sh train.sh`

# 🚜 推理
1. 使用文件：`infer_lora.ipynb`


# 😱 血的教训
1. 一定要从`huggingface`上把[`chatglm-v2-6b`的所有文件](https://huggingface.co/THUDM/chatglm2-6b/tree/main)都下载下来，放在一个文件夹下；这样即使他更新了，也不会影响到你。如果你不下载，你会很被动😒