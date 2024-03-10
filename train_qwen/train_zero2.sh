# --nnodes 1 --nproc_per_node 4 --master_port 25641

deepspeed --include localhost:0 train_qwen2_sft.py \
    --deepspeed ds_zero2_no_offload.json \
    --model_name_or_path model/Qwen1.5-1.8B-Chat \
    --use_lora true \
    --use_deepspeed true \
    --data_path data/custom_data001 \
    --bf16 false \
    --fp16 true \
    --output_dir output_0309_1dot8b \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --report_to "tensorboard" \
    --learning_rate 4e-4 \
    --logging_steps 10 \
    --tf32 False \
    --model_max_length 2048

    # --save_strategy "steps" \
    # --save_steps 10 \ 
    # --save_steps 1000 \
