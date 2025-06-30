import logging
import os
import time
import warnings

import ray
import torch

warnings.filterwarnings("ignore")

from typing import List, Tuple

from accelerate import PartialState
from transformers import AutoModelForCausalLM, AutoTokenizer
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register  # noqa: E402
from verl.single_controller.ray.base import (  # noqa: E402
    RayClassWithInitArgs,
    RayResourcePool,
    RayWorkerGroup,
)
from verl.utils.device import (  # noqa: E402
    get_device_name,
    get_nccl_backend,
)

ray.init()


device_name = get_device_name()


def dispatch_dp_compute(worker_group, *args, **kwargs):
    from verl.single_controller.base.worker_group import WorkerGroup

    assert isinstance(worker_group, WorkerGroup)
    world_size = worker_group.world_size

    # Process args - split each arg list across workers
    split_args = []
    for arg in args:
        if isinstance(arg, (list, tuple)):
            # Split the list into world_size chunks
            chunks = [[] for _ in range(world_size)]
            for i, item in enumerate(arg):
                chunks[i % world_size].append(item)
            split_args.append(tuple(chunks))
        else:
            # If not a list/tuple, duplicate it for each worker
            split_args.append(tuple([arg] * world_size))

    # Process kwargs - split each kwarg list across workers
    split_kwargs = {}
    for k, v in kwargs.items():
        if isinstance(v, (list, tuple)):
            # Split the list into world_size chunks
            chunks = [[] for _ in range(world_size)]
            for i, item in enumerate(v):
                chunks[i % world_size].append(item)
            split_kwargs[k] = tuple(chunks)
        else:
            # If not a list/tuple, duplicate it for each worker
            split_kwargs[k] = tuple([v] * world_size)

    return tuple(split_args), split_kwargs


def collect_dp_compute(worker_group, output):
    from verl.single_controller.base.worker_group import WorkerGroup

    assert isinstance(worker_group, WorkerGroup)
    # Check that we have output from each worker
    assert len(output) == worker_group.world_size

    # Merge all outputs into a single list
    merged_output = []
    for worker_output in output:
        if isinstance(worker_output, list):
            merged_output.extend(worker_output)
        else:
            merged_output.append(worker_output)

    return merged_output


@ray.remote
class TestAccelerateWorker(Worker):
    def __init__(self):
        super().__init__()

        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        self.distributed_state = PartialState(
            backend=f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}",
            rank=rank,
            world_size=world_size,
            init_method=os.environ.get("DIST_INIT_METHOD", None),
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def show_info(self):
        info = {
            "acc_device": self.distributed_state.device,
            "rank": self.rank,
            "world_size": self.world_size,
            "acc_world_size": str(self.distributed_state),
        }
        return info

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_model(self, model_name: str):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=self.distributed_state.device,
            torch_dtype=torch.float16,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Need to set the padding token to the eos token for generation
        self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.model.device

    def _infer(self, prompts: list[str]):
        def formmat_prompt_func(prompt: str):
            messages = [
                {
                    "role": "system",
                    "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
                },
                {"role": "user", "content": prompt},
            ]

            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return text

        res = []
        for batch_simple in prompts:
            # Move the batch to the device
            batch = formmat_prompt_func(batch_simple)
            model_inputs = self.tokenizer([batch], return_tensors="pt").to(
                self.distributed_state.device
            )

            generated_ids = self.model.generate(**model_inputs, max_new_tokens=512)
            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0][:10]
            res.append({"query": batch_simple, "response": response})
        return res

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def infer(self, prompt: str | list[str]):
        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = prompt

        # Split prompts across workers
        splits = [[] for _ in range(self.world_size)]
        for i, prompt in enumerate(prompts):
            splits[i % self.world_size].append(prompt)

        # Get this worker's portion of prompts
        split_world_prompts = splits[self.rank]

        return self._infer(split_world_prompts)

    @register(
        dispatch_mode={
            "dispatch_fn": dispatch_dp_compute,
            "collect_fn": collect_dp_compute,
        }
    )
    def infer_custom(self, prompt: str | list[str]):
        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = prompt

        prompts = prompts[self.rank]

        res = self._infer(prompts)
        return res


resource_pool = RayResourcePool([2], use_gpu=True)


class_with_args = RayClassWithInitArgs(cls=TestAccelerateWorker)
worker_group = RayWorkerGroup(resource_pool, class_with_args)

show_info = worker_group.show_info()

# for i in show_info:
#     print(i)

model_name = "/home/yuanz/documents/weights/Qwen/Qwen2.5-0.5B-Instruct"

model_device = worker_group.load_model(model_name=model_name)


query_list = [
    "你是谁",
    "1+1=几",
    "十个字介绍一下杭州",
]


# way 1

# response_list = worker_group.infer(prompt=query_list)

# print(response_list)


# way 2
response_list = worker_group.infer_custom(prompt=query_list)

for i in response_list:
    print(i)
# time.sleep(20)

ray.shutdown()
