## 使用lora训练thuglm模型
1. `thuglm-6b`模型，最近太火了，而且在中文语言的表现上，效果非常好[https://github.com/THUDM/ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B)，使用int8还可以在小显存上进行推理，非常amazing。
2. 但是目前，训练一个`thuglm-6b`模型，还是比较费劲的（我还没试过，目前都在传使用lora方法来进行训练）。那也就跟风写一个教程。
3. 文本，将介绍如何使用`peft`包（这个包实现了`lora`算法）、对`thuglm-6b`进行微调。
4. 硬件设备是3090（显存为24G）。
5. 包括数据整理、模型转换、训练加载等详细步骤。

完整内容等我下班后，进行补充～


