deepspeed --include localhost:4,5,6,7 example_grpo.py \
    --deepspeed ds_zero2_no_offload.json
