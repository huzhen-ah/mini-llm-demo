# PyTorch Mini LLM

这是一个用于学习和教学的 mini LLM PyTorch 实现，与 `keras-mini-llm/` 保持相同的核心计算链路。项目不依赖 Hugging Face Transformers，从 Byte-level BPE tokenizer 开始，逐步实现 decoder-only Transformer 预训练、LoRA-SFT、LoRA-DPO，以及带 KVCache 的 Prefill / Decode 推理。

> 本项目的目标是展示完整链路和关键实现，不是训练可用于生产环境的通用大语言模型。

## 完整链路

```text
原始文本
  -> Byte-level BPE tokenizer
  -> decoder-only Transformer 预训练
  -> LoRA-SFT 指令微调
  -> LoRA-DPO 偏好优化
  -> Prefill / Decode
  -> KVCache 自回归生成
```

核心组件包括：

- Byte-level BPE 训练、编码和解码
- RMSNorm、RoPE、SwiGLU、causal self-attention
- Pre-Norm Transformer Block 与 tied embedding
- next-token prediction、PAD mask、token accuracy
- LoRA 参数注入、冻结、独立保存和权重合并
- response-only SFT loss
- chosen/rejected DPO loss 与 reference log-probability 预计算
- Prefill / Decode 模型拆分和 KVCache 原地更新
- 不同 prompt 长度的 batch 推理与完成样本动态裁剪

## 文档

Keras 与 PyTorch 共用的实现文档统一位于仓库根目录 `docs/`：

- [Mini LLM 模型架构](../docs/model_architecture.md)
- [Multi-Head Self-Attention 原理与实现](../docs/attention.md)
- [BBPE 原理与实现](../docs/bbpe.md)
- [RoPE 原理与实现](../docs/rope.md)
- [KVCache 推理机制](../docs/kvcache.md)
- [LoRA-SFT 原理与实现](../docs/lora_sft.md)
- [LoRA-DPO 原理与实现](../docs/lora_dpo.md)
- [端到端 demo](../docs/demo.md)
- [推理主流程](../docs/inference.md)

## 环境要求

建议使用 Python 3.10 或更高版本。代码依次选择 NVIDIA CUDA、Apple MPS 和 CPU。

进入 PyTorch 子项目后安装依赖：

```bash
cd pytorch-mini-llm
python -m pip install torch numpy regex tqdm charset-normalizer
```

如需 GPU 版本的 PyTorch，请根据本机 CUDA 环境安装匹配的 PyTorch wheel。

所有脚本都使用相对路径，因此后续命令应在 `pytorch-mini-llm/` 目录内执行。

## 快速开始

仓库已附带 tokenizer 配置、预训练权重、SFT/DPO 合并权重和示例数据。无需重新训练即可运行 KVCache 推理：

```bash
cd pytorch-mini-llm
python interface.py
```

`interface.py` 默认加载：

```text
lora_dpo_weights/0_k2v_lora_merged_weights.pt
```

若要切换为 base 或 SFT 后的模型，请修改文件底部的 `configs`。合并权重已经不再需要 LoRA 分支，因此可以保持 `use_lora=False`，并将 `weight_map_path` 指向：

```text
models/0_k2v_weights.pt
lora_sft_weights/0_k2v_lora_merged_weights.pt
```

可以直接替换 `text_1` 至 `text_4` 或 `prompts` 列表来测试自己的输入。

## 从头运行完整流程

### 方式一：端到端 Demo

```bash
python demo.py
```

`demo.py` 依次检查并运行 BBPE、预训练、LoRA-SFT、LoRA-DPO 和 KVCache 推理。对应产物已经存在时会跳过该训练阶段，因此仓库默认状态下主要用于验证完整链路和推理。

如需真正从头训练，应先备份或移走对应产物，再运行 Demo：

```text
tokenizer_config/vocab.json
tokenizer_config/merge_rules.json
models/0_k2v_weights.pt
lora_sft_weights/0_k2v_lora_merged_weights.pt
lora_dpo_weights/0_k2v_lora_merged_weights.pt
```

### 方式二：分阶段运行

#### 1. 训练 BBPE tokenizer

```bash
python bbpe_trainer.py
```

默认读取 `data/*.txt`，并生成：

```text
tokenizer_config/vocab.json
tokenizer_config/merge_rules.json
```

重新训练 tokenizer 会改变 vocabulary 和 token ID 映射。此后必须重新训练模型，旧 checkpoint 不再兼容。

#### 2. 预训练

```bash
python pretrain.py
```

默认配置：

| 参数 | 默认值 |
| --- | ---: |
| Transformer blocks | 4 |
| Attention heads | 2 |
| Embedding dim | 64 |
| SwiGLU hidden channels | 128 |
| Context size | 200 |
| Batch size | 128 |
| Epochs | 1 |
| Learning rate | 0.001 |

训练目标为 next-token prediction：

```text
X = tokens[:-1]
Y = tokens[1:]
```

loss 和 accuracy 均忽略 `<pad>`。每个 epoch 结束时会执行验证和 Top-k 采样，并保存：

```text
models/{epoch}_k2v_weights.pt
```

仓库当前附带 `models/0_k2v_weights.pt`。

#### 3. LoRA-SFT

```bash
python lora_sft.py
```

默认读取：

```text
base checkpoint: models/0_k2v_weights.pt
training data:   SFT_data/sft_data.jsonl
```

数据采用 messages JSONL 格式。数据管线在 batch 内动态 padding，并只对 assistant response 和结尾 EOS 计算 loss 与 accuracy：

```text
完整序列 + response mask
  -> 动态 padding
  -> X = token_ids[:, :-1]
  -> labels = token_ids[:, 1:]
  -> loss_mask = response_mask[:, 1:]
```

训练时冻结 base model，只优化名称中包含 `lora_` 的参数。默认训练 1 个 epoch，保存两类权重：

```text
lora_sft_weights/0_lora_weights.pt             # 仅 LoRA 参数
lora_sft_weights/0_k2v_lora_merged_weights.pt  # 合并后的完整推理权重
```

#### 4. LoRA-DPO

```bash
python lora_dpo.py
```

默认读取：

```text
base checkpoint: lora_sft_weights/0_k2v_lora_merged_weights.pt
training data:   DPO_data/dpo_data.jsonl
```

每条 DPO 数据包含 chosen 和 rejected response。训练前先用冻结的 SFT 模型预计算 reference log-probability，再创建新的 LoRA 参数进行偏好优化。默认 `beta=0.1`、训练 1 个 epoch，并保存：

```text
lora_dpo_weights/0_lora_weights.pt
lora_dpo_weights/0_k2v_lora_merged_weights.pt
```

#### 5. KVCache 推理

```bash
python interface.py
```

推理分为两个阶段：

```text
完整 prompt
  -> Prefill：并行处理 prompt，创建每层 K/V cache
  -> 采样第一个 token
  -> Decode：每次只处理一个新 token，原地更新 cache
```

整体 cache 布局为：

```text
[layer, batch, head, max_time, head_dim]
```

每个 Decode Transformer Block 只接收自己所在层的：

```text
[batch, head, max_time, head_dim]
```

生成过程中，已遇到 `<eos>` 或达到最大长度的样本会从活动 batch 和 KVCache 中同步移除。

## 权重关系

```text
models/0_k2v_weights.pt
  -> 加载 base 权重并训练 SFT LoRA
  -> lora_sft_weights/0_lora_weights.pt
  -> merge
  -> lora_sft_weights/0_k2v_lora_merged_weights.pt
  -> 作为 DPO base，训练新的 DPO LoRA
  -> lora_dpo_weights/0_lora_weights.pt
  -> merge
  -> lora_dpo_weights/0_k2v_lora_merged_weights.pt
```

模型结构、tokenizer 和 checkpoint 必须匹配。修改 `num_block`、`num_head`、`embedding_dim` 或重新训练 tokenizer 后，不应继续加载原有权重。

## 数据说明

### 预训练数据

`data/` 中包含 15 部武侠作品文本，作为 BBPE tokenizer 与 next-token prediction 预训练语料。PyTorch 与 Keras 版本使用相同的数据集合；替换文本后必须重新训练 tokenizer 和全部后续 checkpoint。

### SFT 数据

`SFT_data/sft_data.jsonl` 共 5200 条，使用 `messages` 格式，训练只监督 assistant response 和结尾 EOS。

### DPO 数据

`DPO_data/dpo_data.jsonl` 共 5200 条，使用 `prompt/chosen/rejected` 格式，为武侠领域回答提供 chosen/rejected 偏好对。

当前数据集中在武侠领域，生成结果会明显偏向相关人物、门派、武功和叙事表达，也可能出现重复退化。结果主要用于验证完整训练链路，不代表模型具备通用中文能力。

## 目录与主要文件

```text
attention.py          # RoPE attention、Prefill attention、Decode attention
bbpe_trainer.py       # Byte-level BPE 训练
callbacks.py          # epoch 结束后的采样与权重保存
demo.py               # 端到端流程
inference_models.py   # Prefill / Decode 模型
interface.py          # KVCache 批量生成接口
layers.py             # RMSNorm 与 SwiGLU
lora_dpo.py           # LoRA-DPO 训练入口
lora_sft.py           # LoRA-SFT 训练入口
lora_utils.py         # LoRA 冻结、保存、加载与合并
losses.py             # 预训练、SFT、DPO loss
metrics.py            # 预训练与 SFT token accuracy
models.py             # 预训练模型
pretrain.py           # 预训练入口
rope.py               # Rotary Position Embedding
sample_utils.py       # Top-k sampling
tokenizer.py          # tokenizer 编码与解码
train_utils.py        # 数据加载、Dataset 与 collate_fn
transformblock.py     # 训练、Prefill、Decode Transformer Block
weight_utils.py       # base state_dict 保存与加载
data/                 # 预训练语料
SFT_data/             # SFT JSONL 数据
DPO_data/             # DPO JSONL 数据
models/               # 预训练 checkpoint
lora_sft_weights/     # SFT LoRA 与 merged checkpoint
lora_dpo_weights/     # DPO LoRA 与 merged checkpoint
tokenizer_config/     # vocabulary 与 merge rules
```

## 与 Keras 版本的主要差异

- Keras `Dense` 与 PyTorch `Linear` 的权重布局和默认初始化不同。
- TensorFlow 通常使用 `int32` token ID；PyTorch cross entropy target 使用 `torch.long`。
- Keras 通过 `compile/fit` 组织训练；PyTorch 显式执行 forward、backward 和 optimizer step。
- Keras KVCache 更新需要构造 scatter indices；PyTorch Decode 直接按 batch/time 索引原地写入。
- PyTorch 全程使用 `[layer, batch, head, time, head_dim]` 的整体 cache 布局。
- 即使计算结构一致，随机初始化和 shuffle 顺序也会令两个版本产生不同的训练曲线。

## 常见问题

### 找不到数据、tokenizer 或 checkpoint

先确认当前工作目录是 `pytorch-mini-llm/`，因为入口脚本使用相对路径。

### 加载权重时出现 size mismatch

检查 tokenizer 和模型结构参数是否与生成该 checkpoint 时一致，重点包括 vocabulary size、block 数、head 数和 embedding dim。

### CPU 训练速度很慢

这是预期现象。可先减小 `batch_size`、`context_size`、`epochs` 或语料规模来验证链路。

### 显存不足

优先减小各入口脚本中的 `batch_size` 和 `context_size`。DPO 会同时处理 chosen/rejected，显存占用通常高于 SFT。

### 生成文本重复或质量较差

当前训练集规模小、题材集中，模型用于代码验证而非质量评测。可扩充许可证明确的多领域语料，并重新训练 tokenizer 和后续全部 checkpoint。

## 当前限制

- 超参数和路径直接写在入口脚本中，尚未提供命令行参数或统一配置文件。
- 尚未实现面向多轮对话的 chat template 和交互式对话入口。
- 未提供分布式训练、混合精度训练、梯度累积和正式 benchmark。
- 当前示例数据与模型规模仅适合教学和链路验证。
