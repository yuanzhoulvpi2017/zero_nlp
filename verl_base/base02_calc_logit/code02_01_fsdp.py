import logging
import os
import time
import warnings

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import ray
import torch

warnings.filterwarnings("ignore")

import json
import logging
import os
import warnings
from dataclasses import asdict
from typing import Optional, Union

import psutil
import torch
import torch.distributed
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register  # noqa: E402
from verl.single_controller.ray.base import (  # noqa: E402
    RayClassWithInitArgs,
    RayResourcePool,
    RayWorkerGroup,
)
from verl.utils import hf_processor, hf_tokenizer, omega_conf_to_dataclass
from verl.utils.activation_offload import enable_activation_offloading
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.debug import DistProfilerExtension, log_gpu_memory_usage
from verl.utils.debug.performance import reduce_timing
from verl.utils.device import (  # noqa: E402
    get_device_id,
    get_device_name,
    get_nccl_backend,
)
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_load_full_state_dict,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
)
from verl.workers.fsdp_workers import create_device_mesh


def set_deterministic():
    # 设置随机种子
    import torch
    import numpy as np
    import os
    import random

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


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

set_deterministic()
device_name = get_device_name()


@ray.remote
class ActorRolloutRefWorker(Worker, DistProfilerExtension):
    def __init__(self):
        Worker.__init__(self)
        set_deterministic()

        import torch.distributed

        if not torch.distributed.is_initialized():
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            torch.distributed.init_process_group(
                backend=f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}",
                rank=rank,
                world_size=world_size,
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )

        # build device mesh for FSDP
        world_size = torch.distributed.get_world_size()
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=-1)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def show_info(self):
        mesh_info = {
            "torch_rank": torch.distributed.get_rank(),
            "torch_world_size": torch.distributed.get_world_size(),
            # "torch_backend": torch.distributed.get_backend(),
            "torch_device_id": get_device_id(),
            "mesh_shape": self.device_mesh.mesh.shape,
            "device_type": self.device_mesh.device_type,
            "is_initialized": torch.distributed.is_initialized(),
        }

        return mesh_info

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self, model_name: str):
        local_path = model_name
        fsdp_config = {
            "param_offload": False,
            "optimizer_offload": False,
            "offload_policy": False,
            "reshard_after_forward": False,
            "forward_prefetch": False,
        }

        self.actor_module_fsdp = self._build_model_optimizer(
            model_path=local_path,
            fsdp_config=fsdp_config,
            enable_gradient_checkpointing=False,
            trust_remote_code=False,
            role="actor",
        )

        return self.actor_module_fsdp.device

    def _build_model_optimizer(
        self,
        model_path,
        fsdp_config,
        enable_gradient_checkpointing=False,
        trust_remote_code=False,
        role="actor",
    ):
        from torch import optim
        from torch.distributed.fsdp import CPUOffload, MixedPrecision
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoModelForVision2Seq,
        )

        from verl.utils.model import (
            get_generation_config,
            print_model_size,
            update_model_config,
        )
        from verl.utils.torch_dtypes import PrecisionType

        assert role in ["actor", "ref"]

        log_gpu_memory_usage(f"Before init {role} from HF AutoModel", logger=logger)
        local_path = model_path

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        # TODO(zhangchi.usc1992): 1. support create from random initialized model. 2. Support init with FSDP directly
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        self.processor = hf_processor(local_path, trust_remote_code=trust_remote_code)

        torch_dtype = fsdp_config.get("model_dtype", None)
        if torch_dtype is None:
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # override model kwargs
        actor_model_config = AutoConfig.from_pretrained(
            local_path,
            trust_remote_code=trust_remote_code,
            attn_implementation="flash_attention_2",
        )

        # patch for kimi-vl
        if getattr(actor_model_config, "model_type", None) == "kimi_vl":
            actor_model_config.text_config.topk_method = "greedy"

        self.generation_config = get_generation_config(
            local_path, trust_remote_code=trust_remote_code
        )

        if self.rank == 0:
            print(f"Model config after override: {actor_model_config}")

        # NOTE(fix me): tie_word_embedding causes meta_tensor init to hang
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not actor_model_config.tie_word_embeddings,
            mesh=self.device_mesh,
        )

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if type(actor_model_config) in AutoModelForVision2Seq._model_mapping.keys():
                actor_module_class = AutoModelForVision2Seq
            else:
                actor_module_class = AutoModelForCausalLM

            actor_module = actor_module_class.from_pretrained(
                pretrained_model_name_or_path=local_path,
                torch_dtype=torch_dtype,
                config=actor_model_config,
                trust_remote_code=trust_remote_code,
            )

            # some parameters may not in torch_dtype. TODO(zhangchi.usc1992) remove this after we switch to fsdp2
            actor_module.to(torch_dtype)
        torch.distributed.barrier()

        if self.rank == 0:
            print_model_size(actor_module)

        log_gpu_memory_usage(f"After init {role} from HF AutoModel", logger=logger)

        # We wrap FSDP for rollout as well
        mixed_precision_config = fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(
                mixed_precision_config.get("param_dtype", "bf16")
            )
            reduce_dtype = PrecisionType.to_dtype(
                mixed_precision_config.get("reduce_dtype", "fp32")
            )

        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32

        auto_wrap_policy = get_fsdp_wrap_policy(
            module=actor_module,
            config=fsdp_config.get("wrap_policy", None),
            is_lora=False,
        )

        if self.rank == 0:
            print(f"wrap_policy: {auto_wrap_policy}")

        fsdp_mesh = self.device_mesh

        cpu_offload = None if role == "actor" else CPUOffload(offload_params=True)

        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            cast_forward_inputs=True,
        )
        if role == "actor" and fsdp_config.get("offload_policy"):
            cpu_offload = CPUOffloadPolicy(pin_memory=True)
            self._is_offload_param = False
            self._is_offload_optimizer = False
        else:
            cpu_offload = None if role == "actor" else CPUOffloadPolicy(pin_memory=True)

        fsdp_kwargs = {
            "mesh": fsdp_mesh,
            "mp_policy": mp_policy,
            "offload_policy": cpu_offload,
            "reshard_after_forward": fsdp_config.get("reshard_after_forward"),
        }
        full_state = actor_module.state_dict()
        apply_fsdp2(actor_module, fsdp_kwargs, fsdp_config)
        fsdp2_load_full_state_dict(actor_module, full_state, fsdp_mesh, cpu_offload)
        actor_module_fsdp = actor_module

        log_gpu_memory_usage(f"After {role} FSDP init", logger=logger)

        return actor_module_fsdp

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def infer(self, prompts: list[str]):
        device_name = get_device_name()
        with torch.no_grad():
            with torch.autocast(device_type=device_name, dtype=torch.bfloat16):
                model_inputs = self.build_model_inputs(
                    prompt=prompts[0], model_device=device_name
                )

                output = self.actor_module_fsdp(**model_inputs)

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

                res = show_model_outputs(output)
                res["prompt"] = prompts

                return res

    def build_model_inputs(self, prompt, model_device):
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
        model_inputs = self.tokenizer([text], return_tensors="pt").to(model_device)
        return model_inputs


if __name__ == "__main__":
    ray.init()
    resource_pool = RayResourcePool([2], use_gpu=True)

    class_with_args = RayClassWithInitArgs(cls=ActorRolloutRefWorker)
    worker_group = RayWorkerGroup(resource_pool, class_with_args)

    show_info = worker_group.show_info()

    # for i in show_info:
    #     print(i)

    # Initialize the model
    model_name_or_path = "/home/yuanz/documents/weights/Qwen/Qwen2.5-0.5B-Instruct"
    model_type = worker_group.init_model(model_name=model_name_or_path)

    print(f"Model type: {model_type}")

    # infer model

    prompts = ["介绍一下你自己"]
    infer_results = worker_group.infer(prompts=prompts)

    for i, result in enumerate(infer_results):
        print(f"Result {i}: {result}")

    time.sleep(20)
    ray.shutdown()
