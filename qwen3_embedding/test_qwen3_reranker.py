# copy code from https://github.com/QwenLM/Qwen3-Embedding/blob/main/examples/qwen3_reranker_transformers.py


import logging
from typing import Dict, Optional, List
import os

import json
import logging
import os
import queue
import sys

from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from tqdm import tqdm
from typing import Union, List, Tuple, Any

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data._utils.worker import ManagerWatchdog

from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoModel,
    is_torch_npu_available,
)

logger = logging.getLogger(__name__)


class Qwen3Reranker:
    def __init__(
        self,
        model_name_or_path: str,
        max_length: int = 4096,
        instruction=None,
        attn_type="causal",
    ) -> None:
        n_gpu = torch.cuda.device_count()
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True, padding_side="left"
        )
        self.lm = (
            AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                # attn_implementation="flash_attention_2",
            )
            .cuda()
            .eval()
        )
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")

        self.prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

        self.prefix_tokens = self.tokenizer.encode(
            self.prefix, add_special_tokens=False
        )
        self.suffix_tokens = self.tokenizer.encode(
            self.suffix, add_special_tokens=False
        )
        self.instruction = instruction
        if self.instruction is None:
            self.instruction = "Given the user query, retrieval the relevant passages"

    def format_instruction(self, instruction, query, doc):
        if instruction is None:
            instruction = self.instruction
        output = (
            "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
                instruction=instruction, query=query, doc=doc
            )
        )
        return output

    def process_inputs(self, pairs):
        out = self.tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=self.max_length
            - len(self.prefix_tokens)
            - len(self.suffix_tokens),
        )
        for i, ele in enumerate(out["input_ids"]):
            out["input_ids"][i] = self.prefix_tokens + ele + self.suffix_tokens
        out = self.tokenizer.pad(
            out, padding=True, return_tensors="pt", max_length=self.max_length
        )
        for key in out:
            out[key] = out[key].to(self.lm.device)
        return out

    @torch.no_grad()
    def compute_logits(self, inputs, **kwargs):
        batch_scores = self.lm(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores

    def compute_scores(self, pairs, instruction=None, **kwargs):
        pairs = [
            self.format_instruction(instruction, query, doc) for query, doc in pairs
        ]
        inputs = self.process_inputs(pairs)
        scores = self.compute_logits(inputs)
        return scores


if __name__ == "__main__":
    model = Qwen3Reranker(
        model_name_or_path="models/Qwen3-Reranker-0.6B",
        instruction="Retrieval document that can answer user's query",
        max_length=2048,
    )
    queries = ["What is the capital of China?", "Explain gravity", "Explain gravity"]
    documents = [
        "The capital of China is Beijing.",
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
    ]
    pairs = list(zip(queries, documents))
    instruction = "Given the user query, retrieval the relevant passages"
    new_scores = model.compute_scores(pairs, instruction)
    print("scores", new_scores)

# scores[0.99853515625, 2.390146255493164e-05, 0.99853515625]
