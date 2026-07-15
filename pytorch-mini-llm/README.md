# PyTorch Mini LLM

这是 `keras-mini-llm/` 的 PyTorch 对照实现，用于学习两个框架在 Tensor API、模型组织、数据加载和训练循环上的差异。

## 当前状态

PyTorch 版本已经实现 decoder-only Transformer 的预训练、LoRA-SFT 和 KVCache 推理主链路：

```text
文本语料
  -> BBPE tokenizer
  -> Dataset / DataLoader
  -> token embedding
  -> Transformer blocks
  -> tied embedding output projection
  -> masked cross entropy
  -> token accuracy
  -> backward / optimizer step
  -> validation
  -> top-k sampling
  -> state_dict checkpoint
  -> LoRA-SFT
  -> prefill
  -> KVCache decode
```

已完成：

- Byte-level BPE trainer 和 tokenizer。
- RMSNorm、RoPE、SwiGLU 和 causal self-attention。
- 多头 Attention 的 reshape / permute 与 padding mask。
- Pre-Norm Transformer Block 和 `nn.ModuleList` 多层堆叠。
- Embedding 与 vocabulary projection 权重共享。
- `Dataset` / `DataLoader` 预训练数据管线。
- 忽略 PAD token 的 cross entropy loss。
- 忽略 PAD token 的 token accuracy。
- 基础 PyTorch 训练循环。
- 验证集 loss 和 token accuracy 统计。
- 训练期间的 Top-k 自回归采样。
- 基于 `state_dict` 的 base model 权重保存和加载。
- Attention 和 SwiGLU 中的 LoRA 层结构及权重合并方法。
- Prefill / decode 模型拆分。
- Prefill 阶段的逐层 K/V 收集。
- Decode 阶段的 KVCache 原地更新。
- 支持不同 prompt 长度和动态 batch 裁剪的推理接口。
- LoRA-only 权重保存、加载和 merged weight 工具。
- SFT JSONL 数据加载、batch 内动态 padding 和 response-only mask。
- 逐 token masked SFT loss 和 SFT token accuracy。
- 冻结 base model、仅优化 LoRA 参数的训练流程。

尚未完成：

- LoRA-DPO 训练流程。
- 端到端 `demo.py`。
- 面向多轮对话的 chat template 和完整对话入口。

## 当前模型

预训练模型包含：

- token embedding
- RMSNorm
- RoPE multi-head causal self-attention
- SwiGLU feed-forward network
- residual connection
- tied embedding output projection

模型输入为 `[batch, sequence]` token IDs，输出为 `[batch, sequence, vocab_size]` logits。

## 运行预训练

进入当前目录后运行：

```bash
python pretrain.py
```

`pretrain.py` 中目前直接定义了模型层数、head 数、embedding 维度、context size、batch size、epoch 和 learning rate，可在学习和调试时直接修改。

训练使用 next-token prediction：

```text
input:  tokens[:-1]
target: tokens[1:]
```

loss 和 accuracy 都会忽略 `pad_id`。

每个 epoch 结束后，当前代码会运行验证、生成一段示例文本，并将不含 LoRA 参数的 base model `state_dict` 保存到 `models/` 目录。仓库当前附带 10 个预训练 checkpoint（`0_k2v_weights.pt` 至 `9_k2v_weights.pt`），可用于继续训练、SFT 和推理实验。

## 运行 LoRA-SFT

进入当前目录后运行：

```bash
python lora_sft.py
```

SFT 数据位于 `SFT_data/emperor_sft_messages_v1.jsonl`，采用 messages JSONL 格式。数据管线先由 `SFTDataset` 返回单条未 padding 的 token IDs 和 response mask，再由 `sft_collate_fn` 在 batch 内动态 padding，并构造 next-token prediction 所需的输入与标签：

```text
完整 token 序列与 response mask
  -> batch 内动态 padding
  -> X = token_ids[:, :-1]
  -> labels = token_ids[:, 1:]
  -> loss_mask = response_mask[:, 1:]
  -> Y = stack(labels, loss_mask)
```

训练前会加载 base model checkpoint，冻结所有非 LoRA 参数，并只将 `requires_grad=True` 的 LoRA 参数交给 Adam。loss 和 accuracy 只统计 assistant response 及其结尾 EOS，不统计 instruction 和 padding。

## KVCache 推理

PyTorch 推理拆分为两个阶段：

```text
完整 prompt
  -> Prefill Model
  -> 初始 K/V cache
  -> 采样第一个 token
  -> Decode Model 逐 token 更新 cache
```

整体 KVCache 使用统一布局：

```text
[layer, batch, head, max_time, head_dim]
```

每个 Decode Transformer Block 只接收当前层的 cache：

```text
[batch, head, max_time, head_dim]
```

Decode Attention 根据每条样本的 `cur_valid_len - 1` 写入当前 token 的 K/V。生成结束的样本会与 KVCache 的 batch 维同步裁剪。

当前推理示例可通过以下命令运行：

```bash
python interface.py
```

`interface.py` 中的模型结构参数和 checkpoint 路径需要与实际预训练产物保持一致。

## 语料与生成效果

仓库当前附带的预训练语料是小规模中国历代帝王人物文本。它适合验证 tokenizer、Transformer、loss、训练、权重保存和生成链路，但不适合训练通用中文生成模型。

当前语料规模小、领域单一，并且“帝”、“王”、“国”等模式频率较高。对领域外 prompt 进行采样时，模型可能出现高频词循环和重复退化。因此，当前生成结果只用于判断代码链路是否跑通，不代表模型已具备实用的语言能力。

后续计划引入许可证明确、题材更丰富的开源中文语料，现有帝王语料可作为小比例领域数据保留。

## 主要文件

```text
attention.py       # RoPE multi-head attention 与 KVCache attention
bbpe_trainer.py    # BBPE 训练
layers.py          # RMSNorm 和 SwiGLU
losses.py          # pretraining loss 与 response-masked SFT loss
metrics.py         # pretraining accuracy 与 response-masked SFT accuracy
models.py          # 预训练模型
inference_models.py # Prefill / decode 推理模型
interface.py       # KVCache 自回归采样入口
lora_utils.py      # LoRA 参数冻结、保存、加载与合并
pretrain.py        # 训练入口和训练循环
lora_sft.py        # LoRA-SFT 训练入口
callbacks.py       # epoch 结束采样与权重保存
rope.py            # RoPE
sample_utils.py    # Top-k sampling
tokenizer.py       # tokenizer 编码与解码
train_utils.py     # 预训练/SFT 数据加载、Dataset 与 SFT collate_fn
transformblock.py  # 预训练、prefill 和 decode Transformer Block
weight_utils.py    # state_dict 保存与加载
data/              # 预训练文本语料
SFT_data/          # messages JSONL 格式的 SFT 数据
models/            # 已提交的预训练 checkpoint
tokenizer_config/  # vocab 和 merge rules
```

## 与 Keras 版本的差异

两个版本的模型计算结构对应，但框架默认值并不完全一致，包括：

- Keras `Dense` 与 PyTorch `Linear` 的权重布局和默认初始化不同。
- Keras 和 PyTorch `Embedding` 的默认初始化不同；当前 PyTorch 版本已将 Embedding 初始化为 `[-0.05, 0.05]` 均匀分布。
- TensorFlow 通常使用 `int32` token IDs，PyTorch cross entropy target 使用 `torch.long`。
- Keras 通过 `compile/fit` 管理训练，PyTorch 显式编写 forward、backward 和 optimizer step。
- Keras KVCache 更新需要构造 scatter indices；PyTorch decode 直接按 batch 和 time 索引原地写入 cache。
- Keras 推理在模型输入输出侧使用 batch-first cache；PyTorch 全程保持 `[layer, batch, head, time, head_dim]`。
- 即使结构一致，两边的随机初始化和 shuffle 顺序也会导致不同的训练曲线。

当前阶段主要目标是完成 PyTorch 版本的逐步翻译和对照学习，不保证两个框架在独立随机初始化下得到相同的 epoch 指标。
