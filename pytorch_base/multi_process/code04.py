import os
from typing import List

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def init_process(rankid, size, func, backend="nccl") -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "65534"  # 295001

    dist.init_process_group(backend=backend, rank=rankid, world_size=size)

    func(rankid, size)


# def run(rank_id, size):
#     tensor = torch.arange(2) + rank_id + 1

#     print(f"before broadcast Rank {rank_id} has data {tensor}")

#     dist.broadcast(tensor, src=0)


#     print(f"after broadcast Rank {rank_id} has data {tensor}")
# def run(rank_id, size):
#     tensor = torch.arange(2, dtype=torch.int64) + 1 + rank_id
#     print("before scatter", " Rank ", rank_id, " has data ", tensor)
#     if rank_id == 0:
#         scatter_list = [
#             torch.tensor([0, 0]),
#             torch.tensor([1, 1]),
#             torch.tensor([2, 2]),
#             torch.tensor([3, 3]),
#         ]
#         print("scater list:", scatter_list)
#         dist.scatter(tensor, src=0, scatter_list=scatter_list)
#     else:
#         dist.scatter(tensor, src=0)
#     print("after scatter", " Rank ", rank_id, " has data ", tensor)
# def run(rank_id, size):
#     tensor = torch.arange(2, dtype=torch.int64) + 1 + rank_id
#     print("before gather", " Rank ", rank_id, " has data ", tensor)
#     if rank_id == 0:
#         gather_list = [torch.zeros(2, dtype=torch.int64) for _ in range(4)]
#         dist.gather(tensor, dst=0, gather_list=gather_list)
#         print("after gather", " Rank ", rank_id, " has data ", tensor)
#         print("gather_list:", gather_list)
#     else:
#         dist.gather(tensor, dst=0)
#         print("after gather", " Rank ", rank_id, " has data ", tensor)


def run(rank_id, size):
    tensor = torch.arange(2, dtype=torch.float64) + 1 + rank_id

    tensor = tensor.to(f"cuda:{rank_id}")
    print("before reudce", " Rank ", rank_id, " has data ", tensor)
    dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
    print("after reudce", " Rank ", rank_id, " has data ", tensor)


if __name__ == "__main__":
    size = 2
    process_list: List[mp.Process] = []

    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()

    for p in process_list:
        p.join()
