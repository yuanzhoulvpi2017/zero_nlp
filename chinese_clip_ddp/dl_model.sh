export HF_ENDPOINT=https://hf-mirror.com


model_list="google/vit-large-patch16-224-in21k BAAI/bge-large-zh-v1.5" 
for tempmodel in $model_list; do
    echo "Download model: $tempmodel"
    huggingface-cli download --resume-download $tempmodel --local-dir "models/$tempmodel" --local-dir-use-symlinks False

done