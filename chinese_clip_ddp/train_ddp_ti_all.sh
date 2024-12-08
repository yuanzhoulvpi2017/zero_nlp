
data_size="large" # small or large
language="zh"

cross_embedding_list="custom base gather_nn gather"


for cross_embedding in $cross_embedding_list; do
    echo "cross_embedding: $cross_embedding"
    TRAINPARAMS="
        --text_model_name_or_path models/BAAI/bge-large-zh-v1.5 \
        --image_model_name_or_path models/image_embedding/image_embedding_001 \
        --data_size $data_size \
        --language $language \
        --cross_embedding $cross_embedding \
        --remove_unused_columns=False \
        --do_train \
        --per_device_train_batch_size=20 \
        --gradient_accumulation_steps 2 \
        --per_device_eval_batch_size=4 \
        --num_train_epochs 2 \
        --warmup_steps=0 \
        --weight_decay 0.1 \
        --overwrite_output_dir \
        --cache_dir cache_dir \
        --save_strategy "steps" \
        --save_steps 1000 \
        --save_total_limit 2 \
        --report_to "tensorboard" \
        --learning_rate 4e-4 \
        --dataloader_pin_memory True \
        --dataloader_num_workers 10 \
        --dataloader_persistent_workers True \
        --logging_steps 10"

    RADOM_PORT=$(shuf -i 1024-65535 -n 1)
    echo "使用的端口号: $RADOM_PORT"
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=1 --nproc-per-node=2 --master_port=$RADOM_PORT train_text_image_embedding.py \
        --output_dir models/train_v1205/model_image-gpu2-$data_size-$language-$cross_embedding \
        $TRAINPARAMS

done
