# --nnodes 1 --nproc_per_node 4 --master_port 25641

deepspeed --include localhost:1,2,3 run.py \
    --deepspeed ds_zero2_no_offload.json \
    --model_name_or_path test_model/model001 \
    --train_type use_lora \
    --data_path data/liuhaotian/LLaVA-CC3M-Pretrain-595K \
    --bf16 true \
    --fp16 false \
    --output_dir output_test_model \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --report_to "tensorboard" \
    --learning_rate 4e-4 \
    --logging_steps 10 
    # --model_max_length 2048

    # --save_strategy "steps" \
    # --save_steps 10 \ 
    # --save_steps 1000 \
    # --save_strategy "epoch" \

