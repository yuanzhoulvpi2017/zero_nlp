from multiprocessing import Process
from typing import List


def func1(x: str) -> str:
    print(f"x value is : {x}")


def main1():
    target_list = [f"value_{i}" for i in range(4)]
    result = [func1(i) for i in target_list]
    print(result)


def main2():
    target_list = [f"value_{i}" for i in range(4)]
    # result = [func1(i) for i in target_list]
    process_list: List[Process] = []

    for i in range(4):
        p = Process(target=func1, args=(target_list[i],))
        process_list.append(p)
        p.start()

    for p in process_list:
        p.join()

    # print(result)


if __name__ == "__main__":
    main2()
