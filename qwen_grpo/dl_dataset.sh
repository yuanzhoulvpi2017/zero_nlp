#!/bin/sh

export HF_ENDPOINT=https://hf-mirror.com

# model_list="Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-14B-Instruct Qwen/Qwen2.5-32B-Instruct Qwen/Qwen2.5-Math-RM-72B"

# model_list="Qwen/Qwen2.5-3B Qwen/Qwen2.5-0.5B" 
# for tempmodel in $model_list; do
#     echo "Processing model: $tempmodel"
#     huggingface-cli download --resume-download $tempmodel --local-dir $tempmodel --local-dir-use-symlinks False

# done

# dataset_name_list="trl-lib/Capybara TIGER-Lab/MathInstruct microsoft/orca-math-word-problems-200k KbsdJames/Omni-MATH meta-math/MetaMathQA AI-MO/NuminaMath-CoT"

dataset_name_list="trl-lib/tldr"
for tempdata in $dataset_name_list; do
    echo "Processing model: $tempmodel"
    huggingface-cli download --resume-download $tempdata --local-dir "data/$tempdata" --repo-type dataset --local-dir-use-symlinks False

done