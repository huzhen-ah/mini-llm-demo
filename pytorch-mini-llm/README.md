# PyTorch Mini LLM

这是 `keras-mini-llm/` 的 PyTorch 对照实现，用于学习两个框架在 Tensor API、模型组织、数据加载和训练循环上的差异。

## 当前状态

PyTorch 版本已经跑通 decoder-only Transformer 的预训练主链路：

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
- Attention 和 SwiGLU 中的 LoRA 层结构及权重合并方法。

尚未完成：

- LoRA-SFT 训练流程。
- LoRA-DPO 训练流程。
- 预训练和 LoRA 权重的完整保存、加载流程。
- Prefill / decode 模型。
- KVCache 推理。
- 完整的采样和对话入口。

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

## 主要文件

```text
attention.py       # RoPE multi-head causal self-attention
bbpe_trainer.py    # BBPE 训练
layers.py          # RMSNorm 和 SwiGLU
losses.py          # padding-aware pretraining loss
metrics.py         # padding-aware token accuracy
models.py          # 预训练模型
pretrain.py        # 训练入口和训练循环
rope.py            # RoPE
tokenizer.py       # tokenizer 编码与解码
train_utils.py     # 语料加载和 Dataset
transformblock.py  # Transformer Block
data/              # 预训练文本语料
tokenizer_config/  # vocab 和 merge rules
```

## 与 Keras 版本的差异

两个版本的模型计算结构对应，但框架默认值并不完全一致，包括：

- Keras `Dense` 与 PyTorch `Linear` 的权重布局和默认初始化不同。
- Keras 和 PyTorch `Embedding` 的默认初始化不同；当前 PyTorch 版本已将 Embedding 初始化为 `[-0.05, 0.05]` 均匀分布。
- TensorFlow 通常使用 `int32` token IDs，PyTorch cross entropy target 使用 `torch.long`。
- Keras 通过 `compile/fit` 管理训练，PyTorch 显式编写 forward、backward 和 optimizer step。
- 即使结构一致，两边的随机初始化和 shuffle 顺序也会导致不同的训练曲线。

当前阶段主要目标是完成 PyTorch 版本的逐步翻译和对照学习，不保证两个框架在独立随机初始化下得到相同的 epoch 指标。
