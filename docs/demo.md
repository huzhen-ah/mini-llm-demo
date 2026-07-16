# 端到端 Demo

Keras / TensorFlow 与 PyTorch 子项目都提供独立的 `demo.py`，用于串联完整工程链路：

```text
BBPE tokenizer
  -> pretrain
  -> LoRA-SFT
  -> LoRA-DPO
  -> merged checkpoint inference
```

两个脚本的阶段、数据格式、模型配置和产物角色一致；训练 API、设备管理及 checkpoint 格式按框架分别实现。

| 实现 | 入口 |
| --- | --- |
| Keras / TensorFlow | `keras-mini-llm/demo.py` |
| PyTorch | `pytorch-mini-llm/demo.py` |

Demo 不会调用其他训练脚本，而是在一个文件中直接组织各阶段所需的模型、数据管线、loss、评估器和权重工具。它用于展示完整链路，不替代各阶段的独立入口。

## 1. 运行目录与数据

两套实现都使用相对于当前子项目的路径，因此必须进入对应目录后运行。

Keras：

```bash
cd keras-mini-llm
python demo.py
```

PyTorch：

```bash
cd pytorch-mini-llm
python demo.py
```

两个 Demo 使用以下相对路径：

```text
data/              # 预训练纯文本
SFT_data/          # sft_data.jsonl
DPO_data/          # dpo_data.jsonl
tokenizer_config/  # vocab.json 与 merge_rules.json
```

这些路径不是每次运行都必须预先齐全：`tokenizer_config/` 可以由 BBPE 阶段创建；某个训练阶段如果被已有 checkpoint 跳过，对应数据在本次运行中就不会读取。若要从头执行完整链路，则需要准备 `data/`、`SFT_data/` 和 `DPO_data/`。

仓库当前已提供武侠小说预训练文本、配套 SFT/DPO 数据和 tokenizer 配置。使用其他数据时，应确保自己拥有相应的使用和分发权利。

## 2. 按产物跳过阶段

`demo.py` 会按 BBPE、Pretrain、LoRA-SFT、LoRA-DPO、Inference 的顺序执行。前四个阶段都会先检查关键产物：

```text
产物存在   -> 跳过该训练阶段
产物不存在 -> 运行该阶段并生成产物
```

这不是断点续训。只要检查文件存在，Demo 就认为该阶段已经完成，不会验证 checkpoint 是否与当前 tokenizer、模型配置或上游权重兼容。

Inference 不会跳过；前置检查完成后总会运行一次示例生成。

## 3. BBPE

两个版本检查相同文件：

```text
tokenizer_config/vocab.json
tokenizer_config/merge_rules.json
```

只要任意一个不存在，就使用 `BBPETrainer(data/*.txt)` 重新执行：

```text
build()
-> train()
-> save()
```

Keras 与 PyTorch 的 BBPE trainer 和 tokenizer 逻辑相同。重新训练 tokenizer 会改变词表和 token ID，因此已有模型 checkpoint 将不再兼容。

## 4. Pretrain

阶段完成标志按框架区分：

```text
Keras  : models/0_k2v_weights.pkl
PyTorch: models/0_k2v_weights.pt
```

共同默认配置为：

```text
num_block       = 4
num_head        = 2
model dimension = 64
hidden_channels = 128
context_size    = 200
batch_size      = 128
epochs          = 1
```

Keras 使用 `model.compile()`、生成器和 `model.fit()`；PyTorch 使用 `PretrainDataset`、`DataLoader` 和显式 forward/backward/optimizer 循环。

训练结束时，评估器执行样例生成并保存 epoch 0 base checkpoint。

## 5. LoRA-SFT

阶段完成标志为：

```text
Keras  : lora_sft_weights/0_k2v_lora_merged_weights.pkl
PyTorch: lora_sft_weights/0_k2v_lora_merged_weights.pt
```

两个版本都会：

```text
1. 创建 use_lora=True 的模型
2. 加载 Pretrain base
3. 冻结非 LoRA 参数
4. 读取 SFT_data/sft_data.jsonl
5. 使用 answer-only mask 训练 LoRA
6. 保存独立 LoRA checkpoint
7. 把 LoRA 合并进 base
8. 保存 SFT merged checkpoint
```

SFT 默认 `batch_size=64`、`context_size=200`、`epochs=1`。Keras 使用生成器和 `model.fit()`；PyTorch 使用 `SFTDataset`、动态 padding collate function 和显式训练/测试循环。

## 6. LoRA-DPO

阶段完成标志为：

```text
Keras  : lora_dpo_weights/0_k2v_lora_merged_weights.pkl
PyTorch: lora_dpo_weights/0_k2v_lora_merged_weights.pt
```

两个版本都会：

```text
1. 创建新的 use_lora=True 模型
2. 加载 SFT merged checkpoint
3. 冻结非 LoRA 参数
4. 读取 DPO_data/dpo_data.jsonl
5. 预计算 reference log-probability
6. 用 chosen/rejected pair 训练 DPO LoRA
7. 保存独立 DPO LoRA checkpoint
8. 合并并保存 DPO merged checkpoint
```

DPO 默认 `batch_size=64`、`context_size=200`、`epochs=1`。两个框架的数据张量定义和 DPO loss 一致，训练 API 与 checkpoint 格式不同。

## 7. Inference

最后阶段设置：

```text
use_lora = False
```

并加载刚检查或生成的 DPO merged checkpoint。因为 LoRA delta 已写入 base 权重，所以推理阶段不再额外加载 LoRA checkpoint。

Keras 创建 `Interface(Tokenizer())`，再把 configs 传给 Prefill/Decode 初始化函数。PyTorch 把 configs 和 `device` 一起传入 `Interface`，随后初始化两个推理模型。

两套脚本最终都会：

```text
示例 prompts
-> BBPE encode
-> Prefill
-> KVCache Decode
-> Top-K Sampling
-> tokenizer.decode
-> 打印生成文本
```

示例 prompts 的具体文字不同，不影响推理流程。

## 8. PyTorch 设备选择

PyTorch Demo 按以下顺序选择设备：

```text
CUDA -> MPS -> CPU
```

并设置：

```python
torch.manual_seed(42)
```

模型、训练 batch、有效长度和 KVCache 都放在选定设备上；用于 NumPy Top-K Sampling 的 logits 会转回 CPU。

Keras Demo 没有在脚本中显式选择设备，由 TensorFlow/Keras 运行环境决定。

## 9. 生成文件

从缺少产物的状态运行 Demo，可能生成：

```text
tokenizer_config/*.json
models/*
lora_sft_weights/*
lora_dpo_weights/*
```

主要格式为：

| 产物 | Keras / TensorFlow | PyTorch |
| --- | --- | --- |
| base/merged 权重映射 | `.pkl` | `.pt` |
| 独立 LoRA 权重 | `.pkl` | `.pt` |
| Keras 训练 checkpoint | 还可能生成 `.weights.h5` | 不适用 |

根 `.gitignore` 默认忽略新的 `.pt`/`.pth`，但明确允许三个 PyTorch 权重目录中的 `.pt`；`.pkl` 和 `.weights.h5` 没有被该规则自动忽略。提交新训练产物前应检查 `git status`。

## 10. 使用限制

- 默认模型很小，只用于验证工程链路，不代表实际生成质量。
- Demo 默认最多训练一个 epoch，重点是跑通流程。
- 文件存在检查不会验证产物内容、shape 或上下游兼容性。
- 如果只删除下游 checkpoint，Demo 会复用仍然存在的上游产物。
- 如果重新训练 tokenizer，应同时重新训练所有模型阶段。
- 如果改变模型结构或词表，仅保留旧 checkpoint 可能在加载时产生缺失参数或 shape mismatch。
- 推理默认使用随机 Top-K Sampling；即使模型权重相同，多次输出也不保证完全一致。

## 11. 完整流程总结

```text
检查 tokenizer 配置
-> 缺失则训练 BBPE
-> 检查 Pretrain base
-> 缺失则预训练
-> 检查 SFT merged base
-> 缺失则训练、保存并合并 SFT LoRA
-> 检查 DPO merged base
-> 缺失则预计算 ref logp、训练并合并 DPO LoRA
-> 加载 DPO merged base
-> Prefill / Decode KVCache 推理
-> 输出文本
```
