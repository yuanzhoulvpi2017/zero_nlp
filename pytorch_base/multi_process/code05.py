import os
from typing import List

import torch
import torch.distributed as dist


def run(rank_id, size):
    tensor = torch.arange(2, dtype=torch.float64) + 1 + rank_id

    tensor = tensor.to(f"cuda:{rank_id}")
    print("---->before reudce", " Rank ", rank_id, " has data ", tensor, "<----")
    dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
    print("---->after reudce", " Rank ", rank_id, " has data ", tensor, "<----")


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    dist.init_process_group(backend="nccl")
    run(rank, size=0)


if __name__ == "__main__":
    main()
