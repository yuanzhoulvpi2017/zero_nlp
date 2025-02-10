#!/bin/sh

export HF_ENDPOINT=https://hf-mirror.com

# model_list="Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-14B-Instruct Qwen/Qwen2.5-32B-Instruct"

model_list="weqweasdas/RM-Gemma-2B" #"OpenGVLab/Mono-InternVL-2B" #"Qwen/Qwen2.5-0.5B-Instruct" #
for tempmodel in $model_list; do
    echo "Processing model: $tempmodel"
    huggingface-cli download --resume-download $tempmodel --local-dir "model/$tempmodel" --local-dir-use-symlinks False

done
