torchrun --nproc_per_node=4 \
    --nnode=2 \
    --node_rank=1 \
    --master_addr="127.0.0.1" \
    --master_port=1234 \
    code06.py
  