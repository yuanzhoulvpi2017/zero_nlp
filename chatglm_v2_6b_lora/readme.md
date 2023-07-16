# 🚀 最简单、最便宜的训练`thu-chatglm-6b`模型教程 🎯 
1. 感谢智谱AI开源`chatglm-v2-6b`大模型；
2. 之前就给`v1`版本做过lora，在智谱AI宣布`v2`可以商用后，打算给`v2`也做一版lora；
3. 基于`v2`的[官网代码](https://github.com/THUDM/ChatGLM2-6B/tree/main/ptuning)，做了简单修改；

## 📝 更新记录
1. **07-14 版本** `chatglm-v2-6b`模型的`lora`训练方案🔗👉[**chatglm_v2_6b_lora**](https://github.com/yuanzhoulvpi2017/zero_nlp/tree/main/chatglm_v2_6b_lora)

# 🔄 训练

## 使用vscode调试
1. 这个已经写好了，就在`.vscode/launch.json`里面；

## 直接使用sh

1. `sh train.sh`

# 🚜 推理
1. 使用文件：`infer_lora.ipynb`


# 😱 血的教训
1. 一定要从`huggingface`上把[`chatglm-v2-6b`的所有文件](https://huggingface.co/THUDM/chatglm2-6b/tree/main)都下载下来，放在一个文件夹下；这样即使他更新了，也不会影响到你。如果你不下载，你会很被动😒