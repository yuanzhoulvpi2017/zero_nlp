## 对qwen2的微调

## 使用的数据

### download from github

```BASH
git lfs clone https://github.com/Toyhom/Chinese-medical-dialogue-data.git
```

### convert data to instruction data

```python
import json
import pandas as pd
from datasets import load_dataset
import os
from pathlib import Path
import shutil
from glob import glob

```

```python

all_file_list = glob("Chinese-medical-dialogue-data/*/*/*.csv")

alldata = (
    pd.concat([pd.read_csv(i, encoding="GB18030") for i in all_file_list])
    .pipe(lambda x: x[["ask", "answer"]])
    .pipe(lambda x: x.loc[x["ask"].apply(lambda j: str(j) != "无")])
    .pipe(lambda x: x.loc[x["ask"].apply(lambda j: len(str(j).replace(" ", "")) > 10)])
    .pipe(
        lambda x: x.loc[x["answer"].apply(lambda j: len(str(j).replace(" ", "")) > 5)]
    )
    .pipe(lambda x: x.loc[x["ask"].apply(lambda j: len(str(j)) <= 100)])
    .pipe(lambda x: x.loc[x["answer"].apply(lambda j: len(str(j)) <= 500)])
    .sample(frac=1.0)
    .reset_index(drop=True)
)

```

```python
from tqdm import tqdm

chunk_size = 100
chunk_s_list = list(range(0, alldata.shape[0], chunk_size))

target_dir = "custom_data001"
shutil.rmtree(target_dir, ignore_errors=True)
os.makedirs(target_dir)

for index, chunk_s in tqdm(enumerate(chunk_s_list), total=len(chunk_s_list)):
    temp_data = alldata.loc[chunk_s: (chunk_s + chunk_size)]
    with open(f"{target_dir}/data_{index}.json", encoding="utf-8", mode="w") as fout:
        for i, iter in temp_data.iterrows():
            fout.write(
                json.dumps(
                    {"instruction": str(iter["ask"]), "output": str(iter["answer"])},
                    ensure_ascii=False,
                )
                + "\n"
            )

```

## 使用的模型

使用的模式为qwen1.5-xxB-chat，具体的下载链接为[Qwen/qwen15](https://huggingface.co/collections/Qwen/qwen15-65c0a2f577b1ecb76d786524)

## 修改的目的
主要是在transformers的Trainer的这个部分代码，出现问题[https://github.com/huggingface/transformers/blob/efdd436663436e78d8ad3213d11325d86578db95/src/transformers/trainer.py#L2401](https://github.com/huggingface/transformers/blob/efdd436663436e78d8ad3213d11325d86578db95/src/transformers/trainer.py#L2401)

#### 原始代码

```python

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            if is_torch_tpu_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)
```

#### 修改之后的代码

代码链接： [https://github.com/yuanzhoulvpi2017/zero_nlp/blob/4fb3e8fb12b24c9d469ca88bee83e764c90bda8b/train_qwen/train_qwen2_sft.py#L258](https://github.com/yuanzhoulvpi2017/zero_nlp/blob/4fb3e8fb12b24c9d469ca88bee83e764c90bda8b/train_qwen/train_qwen2_sft.py#L258)
```python
            if grad_norm is not None:
                logs["grad_norm"] = (
                    grad_norm
                    if not isinstance(grad_norm, torch.Tensor)
                    else grad_norm.detach().item()
                )

```