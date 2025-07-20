import logging
import os
import time
import warnings

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import ray
import torch

warnings.filterwarnings("ignore")

import random
from typing import List, Tuple
import os
import numpy as np
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


def set_deterministic():
    # 设置随机种子
    import torch
    import numpy as np
    import os

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # 设置CUDA确定性计算
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 设置CUDA随机种子
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)


# 在Ray worker中调用
set_deterministic()
ray.init()


device_name = get_device_name()


@ray.remote
class TestAccelerateWorker(Worker):
    def __init__(self):
        super().__init__()
        set_deterministic()

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

            with torch.no_grad():
                output = self.model(
                    **model_inputs,
                )

            def show_model_outputs(model_outputs):
                logits = model_outputs.logits
                logits_info = {
                    "device": logits.device,
                    "shape": logits.shape,
                    "dtype": logits.dtype,
                    "top_tokens": torch.argmax(logits, dim=-1)[
                        0, :3
                    ].tolist(),  # Convert to list for better readability
                    "logits_mean": logits.mean(axis=1)[
                        0, :3
                    ].tolist(),  # Convert to list for better readability
                    "value": logits[
                        0, 0, :10
                    ].tolist(),  # Convert to list for better readability
                    "grad_enabled": logits.requires_grad,
                }
                return logits_info

            res.append({**show_model_outputs(output), "query": batch_simple})
        return res

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def infer(self, prompt: str | list[str]):
        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = prompt
        # Get this worker's portion of prompts
        split_world_prompts = prompts

        return self._infer(split_world_prompts)

    @register(
        dispatch_mode=Dispatch.ONE_TO_ALL,
    )
    def infer_custom(self, prompt: str | list[str]):
        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = prompt

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


query_list = ["介绍一下你自己"]


# way 1

# response_list = worker_group.infer(prompt=query_list)

# print(response_list)


# way 2
response_list = worker_group.infer_custom(prompt=query_list)

for i in response_list:
    print(i)
# time.sleep(20)

ray.shutdown()
