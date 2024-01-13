# --nnodes 1 --nproc_per_node 4 --master_port 25641

deepspeed --include localhost:0,1,2,3 train_sft.py \
    --deepspeed ds_zero2_no_offload.json \
    --model_name_or_path internlm-7b \
    --use_lora true \
    --use_deepspeed true \
    --data_path hz_sft_datav2 \
    --bf16 true \
    --fp16 false \
    --output_dir output_refusev2 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 3 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --learning_rate 4e-4 \
    --logging_steps 10 \
    --tf32 False \
    --model_max_length 2048

# --save_steps 1000 \
