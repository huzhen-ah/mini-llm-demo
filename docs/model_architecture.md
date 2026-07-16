# Mini LLM 模型架构

本文说明项目中 Keras / TensorFlow 与 PyTorch 两个版本共用的模型架构。两套实现使用相同的数据流和核心计算，只在框架 API、参数命名及模型组织方式上有所区别。

对应代码主要在：

| 模块 | Keras / TensorFlow | PyTorch |
| --- | --- | --- |
| 模型组装 | `keras-mini-llm/models.py` | `pytorch-mini-llm/models.py` |
| Transformer Block | `keras-mini-llm/transformblock.py` | `pytorch-mini-llm/transformblock.py` |
| Attention | `keras-mini-llm/attention.py` | `pytorch-mini-llm/attention.py` |
| RMSNorm、SwiGLU | `keras-mini-llm/layers.py` | `pytorch-mini-llm/layers.py` |
| RoPE | `keras-mini-llm/rope.py` | `pytorch-mini-llm/rope.py` |

本文聚焦训练模型 `create_pretrain_model()`。Prefill、Decode 和 KVCache 属于推理阶段的结构变化，另见现有 KVCache 文档。

## 1. 整体结构

本项目实现的是 decoder-only Transformer。输入 token IDs 依次经过 embedding、多个 Transformer Block 和词表投影，最终得到每个位置预测下一个 token 的 logits：

```text
token IDs
  -> token embedding
  -> Transformer Block × N
       -> RMSNorm
       -> causal multi-head self-attention + RoPE
       -> residual connection
       -> RMSNorm
       -> SwiGLU FFN
       -> residual connection
  -> embedding 权重转置投影
  -> logits
```

模型没有 encoder，也没有 cross-attention。每个位置只能关注自己和此前的位置，因此适合自回归生成。

## 2. 配置参数

`create_pretrain_model(configs)` 从配置字典读取以下参数：

| 含义 | Keras 名称 | PyTorch 名称 |
| --- | --- | --- |
| Transformer Block 数量 | `num_block` | `num_block` |
| attention head 数量 | `num_head` | `num_head` |
| 模型维度 | `embedding_size` | `embedding_dim` |
| SwiGLU 中间维度 | `hidden_channels` | `hidden_channels` |
| 词表大小 | `vocab_size` | `vocab_size` |
| padding token ID | `pad_id` | `pad_id` |
| 是否启用 LoRA | `use_lora`，默认 `False` | `use_lora`，默认 `False` |

当前预训练脚本的默认小模型配置是：

```text
num_block      = 4
num_head       = 2
model_dim      = 64
hidden_channels = 128
context_size   = 200
```

其中模型维度必须能被 head 数量整除：

```text
head_dim = model_dim / num_head
```

## 3. 输入、标签与输出

设：

```text
B : batch size
T : 序列长度
D : 模型维度
V : 词表大小
```

模型输入、隐藏状态和输出的形状为：

```text
input_ids : (B, T)
hidden    : (B, T, D)
logits    : (B, T, V)
```

预训练数据会先在序列末尾追加 `<eos>`，再错开一位构造输入和标签：

```text
完整序列：x_0, x_1, x_2, ..., x_n, <eos>
模型输入：x_0, x_1, x_2, ..., x_n
训练标签：x_1, x_2, x_3, ..., <eos>
```

因此位置 `t` 的 logits 用来预测位置 `t + 1` 的 token。模型直接输出 logits，不在模型末尾执行 softmax；交叉熵损失内部会完成相应计算。

## 4. Token Embedding

输入的 token ID 通过可训练的 embedding 表转换为向量：

```text
E ∈ R^(V × D)
H_0 = E[input_ids]
```

形状变化为：

```text
(B, T) -> (B, T, D)
```

项目没有把绝对位置向量加到 token embedding 上。位置信息在 attention 内通过 RoPE 注入 query 和 key。

## 5. Padding Mask

模型根据 `pad_id` 在内部创建 padding mask：

```text
padding_mask = (input_ids == pad_id)
```

mask 的形状是 `(B, T)`，约定如下：

```text
0 : 有效 token
1 : padding token
```

它会传入每个 Transformer Block，并最终与 causal mask 合并。当前 attention 使用它屏蔽 padding key，使有效 query 不会读取 padding 位置。

预训练数据通常由固定长度片段组成，基本不需要 padding；SFT、DPO 或批量推理中序列长度不同时，这个 mask 才更重要。

## 6. Transformer Block

每个 Block 采用 pre-norm 结构：先归一化，再执行子层计算，最后做残差相加。

设输入为 `x`，一个 Block 的计算可写为：

```text
h = x + Attention(RMSNorm(x))
y = h + SwiGLU(RMSNorm(h))
```

与 post-norm 相比，这里的 RMSNorm 位于 attention 和 FFN 之前。每个 Block 包含两套独立的 RMSNorm 参数。

多个 Block 顺序堆叠：

```text
H_i = Block_i(H_(i-1))
```

所有 Block 的结构相同，但参数互不共享。

## 7. RMSNorm

RMSNorm 对每个 token 的最后一个维度进行归一化。对向量 `x ∈ R^D`：

```text
rms(x) = sqrt(mean(x²) + epsilon)
RMSNorm(x) = x / rms(x) * gamma
```

当前实现中：

```text
epsilon = 1e-6
gamma   : 形状为 (D,) 的可训练缩放参数
```

实现没有减去均值，也没有 bias，这正是 RMSNorm 与 LayerNorm 的主要区别之一。

## 8. Causal Multi-Head Self-Attention

### 8.1 Q、K、V 投影

归一化后的隐藏状态分别经过三个线性层：

```text
Q = X W_q + b_q
K = X W_k + b_k
V = X W_v + b_v
```

Q、K、V 的形状起初都是 `(B, T, D)`，随后拆分为多个 head：

```text
(B, T, D) -> (B, H, T, Dh)
```

其中 `H = num_head`，`Dh = D / H`。

### 8.2 RoPE

项目只对 Q 和 K 应用 RoPE：

```text
Q_rot = RoPE(Q)
K_rot = RoPE(K)
```

V 不做旋转。RoPE 让 attention score 能反映 token 之间的相对位置。其数学推导和代码细节见 `keras-mini-llm/docs/rope.md`；该原理同样适用于 PyTorch 实现。

### 8.3 Attention Score 与 Mask

每个 head 的 attention score 为：

```text
S = Q_rot K_rot^T / sqrt(Dh)
```

形状为：

```text
(B, H, T, T)
```

模型同时使用两类 mask：

- causal mask：屏蔽当前位置右侧的未来 token。
- padding mask：屏蔽作为 key 的 padding token。

被屏蔽的位置在 softmax 前加上一个很大的负数 `-1e10`。随后：

```text
A = softmax(S + mask)
O = A V
```

各个 head 的结果重新拼接为 `(B, T, D)`，再经过输出投影 `W_o`，最后与 Block 输入做残差相加。

## 9. SwiGLU FFN

Attention 子层之后是 SwiGLU 前馈网络。项目使用两个并行的输入投影：

```text
xv = W_v x
xw = W_w x
gate = SiLU(xw) = sigmoid(xw) * xw
hidden = gate * xv
out = W_o hidden
```

形状变化为：

```text
(B, T, D)
  -> 两路 (B, T, hidden_channels)
  -> 逐元素相乘
  -> (B, T, D)
```

三个投影都不使用 bias。FFN 输出保持模型维度不变，因此可以与子层输入做残差相加。

## 10. 输出投影与权重绑定

最后一个 Transformer Block 输出 `(B, T, D)` 的隐藏状态。项目没有另建独立的语言模型输出层，而是复用 token embedding 权重的转置：

```text
logits = hidden E^T
```

形状变化为：

```text
(B, T, D) × (D, V) -> (B, T, V)
```

这种做法称为 weight tying。输入 embedding 与输出词表投影共享同一份参数，可以减少参数量，并让 token 的输入表示与输出分类空间保持关联。输出投影不包含额外 bias。

## 11. LoRA 在架构中的位置

`use_lora=False` 时，模型只使用基础线性层。启用 LoRA 后，项目会在以下投影旁增加低秩增量分支：

- Attention：Q、K、V 和输出投影。
- SwiGLU：两路输入投影和输出投影。

基础形式为：

```text
y = W x + scale * B(A(x))
scale = alpha / rank
```

LoRA 的 B 矩阵初始化为零，因此刚启用时增量分支输出为零，不会立即改变基础模型结果。训练完成后，增量权重可以合并回基础权重。

LoRA 改变的是线性投影的参数更新方式，不改变主干模型的数据流和张量形状。

## 12. Keras 与 PyTorch 的实现差异

两套训练模型在数学结构上相同，主要区别如下：

| 项目 | Keras / TensorFlow | PyTorch |
| --- | --- | --- |
| 模型组织 | Keras Functional Model | `nn.Module` 内部类 `Pretrain_Model` |
| Block 容器 | 构图时逐层调用 | `nn.ModuleList` |
| 前向方法 | `call()` | `forward()` |
| 模型维度配置名 | `embedding_size` | `embedding_dim` |
| Embedding 权重 | `embedding_layer.embeddings` | `self.embedding.weight` |
| 训练入口 | `model.compile()` / `model.fit()` | 显式 optimizer、backward 和训练循环 |

这些差异不改变模型本身的计算定义，因此架构说明可以共用。涉及 Prefill / Decode 的 KVCache 时，两套实现的 cache 形状和逐层传递方式存在差异，应在 KVCache 文档中分别说明。

## 13. 完整前向流程

把以上模块合在一起，训练模型的一次前向计算可以概括为：

```text
1. 根据 input_ids 创建 padding mask
2. 查 embedding 表，得到 (B, T, D)
3. 依次通过 N 个 Transformer Block
   3.1 RMSNorm
   3.2 Q/K/V 投影并拆分多头
   3.3 对 Q/K 应用 RoPE
   3.4 应用 causal mask 与 padding mask
   3.5 attention 输出投影与残差连接
   3.6 RMSNorm
   3.7 SwiGLU FFN 与残差连接
4. 使用 embedding 权重转置投影到词表
5. 返回 (B, T, V) logits
6. 使用错开一位的标签计算 next-token loss
```

训练、SFT、DPO 和普通非 KVCache 推理都复用这套 decoder-only 主干；它们的主要区别在数据构造、损失函数以及哪些参数参与训练。
