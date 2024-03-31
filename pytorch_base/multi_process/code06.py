import os
from typing import List

import torch
import torch.distributed as dist


def run(local_rank: int, rank: int) -> None:
    print(f"local_rank : {local_rank},  rank : {rank} \n")


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    dist.init_process_group(backend="nccl")
    run(local_rank, rank)


if __name__ == "__main__":
    main()
