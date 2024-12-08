export HF_ENDPOINT=https://hf-mirror.com



dataset_name_list="jackyhate/text-to-image-2M"
for tempdata in $dataset_name_list; do
    echo "download dataset: $tempmodel"
    huggingface-cli download --resume-download $tempdata --local-dir "data/$tempdata" --repo-type dataset --local-dir-use-symlinks False

done