import os
from typing import List

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def init_process(rankid, size, func, backend="gloo") -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "65534"  # 295001

    dist.init_process_group(backend=backend, rank=rankid, world_size=size)

    func(rankid, size)


def run(rank_id, size):
    tensor = torch.zeros(1)
    if rank_id == 0:
        tensor += 1
        # Send the tensor to process 1
        dist.send(tensor=tensor, dst=1)
        print("after send, Rank ", rank_id, " has data ", tensor[0])

        dist.recv(tensor=tensor, src=1)
        print("after recv, Rank ", rank_id, " has data ", tensor[0])
    else:
        # Receive tensor from process 0
        dist.recv(tensor=tensor, src=0)
        print("after recv, Rank ", rank_id, " has data ", tensor[0])

        tensor += 1
        dist.send(tensor=tensor, dst=0)
        print("after send, Rank ", rank_id, " has data ", tensor[0])


if __name__ == "__main__":
    size = 2
    process_list: List[mp.Process] = []

    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()

    for p in process_list:
        p.join()


        
