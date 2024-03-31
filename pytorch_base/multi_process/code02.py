# from multiprocessing import Process
from torch.multiprocessing import Process
from typing import List
import torch
import os


# def func1(x: str) -> str:
#     print(f"x value is : {x}")


def func1(x: torch.Tensor) -> None:
    # veryyyy complete  
    x2 = x + 20.0
    print(f"pid : {os.getpid()} x2 shape is : {x2.shape}")


def main2():
    target_list = [torch.randint(0, 10, size=(i + 1, 1)) for i in range(6)]
    # result = [func1(i) for i in target_list]
    process_list: List[Process] = []

    for i in range(len(target_list)):
        p = Process(target=func1, args=(target_list[i],))
        process_list.append(p)
        p.start()

    for p in process_list:
        p.join()

    # print(result)


if __name__ == "__main__":
    main2()
