# Multi-Head Self-Attention 原理与实现

本文说明项目中普通训练模型使用的 causal multi-head self-attention。Keras / TensorFlow 与 PyTorch 两个版本的数学过程和张量布局一致，因此共用一篇文档。

对应代码主要在：

- `keras-mini-llm/attention.py` 中的 `AttentionWithRoPE`
- `pytorch-mini-llm/attention.py` 中的 `AttentionWithRoPE`
- 两个版本各自的 `rope.py`
- 两个版本各自的 `transformblock.py`

本文讨论的是一次处理完整序列的普通 Attention。Prefill 和单 token Decode 会额外读写 KVCache，虽然核心注意力公式相同，但缓存形状和数据传递方式不同，应放在 KVCache 文档中分别说明。

## 1. Attention 在模型中的位置

每个 Transformer Block 采用 pre-norm 结构：

```text
x
  -> RMSNorm
  -> Multi-Head Self-Attention with RoPE
  -> 与 x 做残差相加
  -> RMSNorm
  -> SwiGLU FFN
  -> 残差相加
```

如果只看 attention 子层，可以写成：

```text
h = x + Attention(RMSNorm(x), mask)
```

Attention 接收整段隐藏状态，让序列中每个位置从允许访问的位置聚合信息。由于项目是 decoder-only 自回归模型，当前位置不能读取未来 token。

## 2. Self-Attention 的输入与输出

设：

```text
B  : batch size
T  : sequence length
D  : model dimension
H  : number of heads
Dh : head dimension，Dh = D / H
```

输入与输出形状相同：

```text
input  x : (B, T, D)
output y : (B, T, D)
```

`D` 必须能被 `H` 整除。当前代码直接用整数除法拆分最后一维，没有单独的参数校验；如果不能整除，reshape 会失败。

这里的 self-attention 表示 query、key、value 都来自同一个输入 `x`。项目没有 cross-attention。

## 3. 从输入生成 Q、K、V

输入先经过三套独立线性投影：

```text
Q = x Wq + bq
K = x Wk + bk
V = x Wv + bv
```

形状保持不变：

```text
x : (B, T, D)
Q : (B, T, D)
K : (B, T, D)
V : (B, T, D)
```

三套投影的作用不同：

- query 表示当前位置希望查找什么信息。
- key 表示每个位置可被匹配的特征。
- value 表示匹配后真正参与聚合的内容。

Keras 使用四个 `Dense(D)`，PyTorch 使用四个 `nn.Linear(D, D)`；除 Q、K、V 外，第四个线性层用于最终输出投影。当前四个基础投影都包含 bias。

## 4. 为什么拆成多个 Head

如果只计算一次宽度为 `D` 的 attention，所有匹配关系都压在同一个注意力空间中。Multi-head attention 把模型维度拆成 `H` 个较小的子空间，让不同 head 可以学习不同的匹配模式。

项目中的形状变化是：

```text
(B, T, D)
  -> reshape (B, T, H, Dh)
  -> transpose / permute (B, H, T, Dh)
```

其中：

```text
Dh = D / H
```

Keras 的辅助方法名是 `reshape_and_transpose()`，PyTorch 对应 `reshape_and_permute()`。函数名不同，但得到的布局完全相同：

```text
(batch, head, time, head_dim)
```

## 5. RoPE 如何进入 Attention

拆分多头之后，项目对 Q 和 K 应用 RoPE：

```text
Q_rot = RoPE(Q)
K_rot = RoPE(K)
```

V 不应用 RoPE。原因是位置信息需要影响 query 与 key 的匹配分数，而 value 负责承载最终聚合的内容。

普通完整序列模式下，位置序号为：

```text
0, 1, 2, ..., T - 1
```

`rope_exp()` 将每个 head 的最后一维分成等长的左右两半，将对应元素组成复数，再根据位置和频率执行复数旋转。因而 `Dh` 还需要是偶数，否则左右两半无法正确配对。

RoPE 不改变张量形状：

```text
(B, H, T, Dh) -> (B, H, T, Dh)
```

RoPE 的完整数学推导见 [RoPE 原理与实现](rope.md)。

## 6. 计算 Attention Score

经过 RoPE 后，对每个 head 计算 query 与所有 key 的点积：

```text
raw_score[b, h, m, n]
  = dot(Q_rot[b, h, m, :], K_rot[b, h, n, :])
```

其中：

```text
m : query 位置
n : key 位置
```

代码使用 einsum：

```text
bhms,bhns->bhmn
```

形状变化为：

```text
Q_rot : (B, H, T, Dh)
K_rot : (B, H, T, Dh)
score : (B, H, T, T)
```

score 的最后两维可以理解为一张 `query × key` 的表。第 `m` 行表示位置 `m` 对所有 key 位置的匹配程度。

## 7. 为什么除以 `sqrt(Dh)`

点积会随着向量维度增大而增大。如果 score 数值过大，softmax 容易过早进入接近 0 或 1 的饱和区间，导致梯度变小。

因此项目在 softmax 前执行缩放：

```text
scaled_score = score / sqrt(Dh)
```

代码中的 `s` 是投影后的总模型维度 `D`，所以：

```text
s // num_head = Dh
```

## 8. Causal Mask

### 8.1 为什么需要 Causal Mask

训练时整段序列会一次送入模型。如果不做限制，位置 `m` 可以直接看到右侧的真实未来 token，next-token prediction 就会发生信息泄漏。

Causal mask 的规则是：

```text
当 n > m 时，屏蔽 key 位置 n
```

也就是当前位置可以读取自己和左侧历史，但不能读取右侧未来。

### 8.2 当前代码如何构造

代码创建 query 和 key 的位置索引：

```text
mask_q : (T, 1)
mask_k : (T,)
```

再比较：

```text
causal_mask[m, n] = 1 if m < n else 0
```

长度为 4 时，mask 是：

```text
key n ->  0  1  2  3
query m
0          0  1  1  1
1          0  0  1  1
2          0  0  0  1
3          0  0  0  0
```

其中 `1` 表示需要屏蔽，`0` 表示可以访问。

这个二维 mask 会扩展为：

```text
(1, 1, T, T)
```

同一份因果关系可广播到所有 batch 和所有 head。

## 9. Padding Mask

模型在 `models.py` 中根据输入 token ID 创建 padding mask：

```text
padding_mask = (input_ids == pad_id)
```

形状与约定为：

```text
shape : (B, T)
0     : 有效 token
1     : padding token
```

传入 attention 后扩展为：

```text
(B, 1, 1, T)
```

这里最后一维对应 key 位置，因此 padding mask 屏蔽的是 padding key。它会广播到所有 head 和所有 query 位置。

项目使用逐元素 `maximum` 合并 causal mask 与 padding mask：

```text
combined_mask = maximum(causal_mask, padding_mask)
```

因为两种 mask 都使用 `1` 表示屏蔽，所以只要任意一个条件成立，该位置就会被屏蔽。这等价于逻辑 OR。

## 10. Mask 如何作用于 Softmax

合并后的 mask 乘以 `-1e10`，再加到原始 score：

```text
masked_score = raw_score + combined_mask * (-1e10)
attention_weight = softmax(masked_score / sqrt(Dh))
```

被屏蔽位置会变成极大的负数，经过 softmax 后权重接近 0。

当没有传入 padding mask 时，代码走另一个分支：

```text
masked_score = raw_score - causal_mask * 1e10
```

两条分支对 causal mask 的效果相同。

需要注意，当前实现先把 `-1e10` 加到 score，再整体除以 `sqrt(Dh)`。因此实际送入 softmax 的屏蔽值是 `-1e10 / sqrt(Dh)`；它仍然足够小，能够把相应权重压到接近 0。

## 11. Padding Query 为什么没有在 Attention 内单独屏蔽

当前 padding mask 只沿 key 维广播，没有把 padding query 的整行输出强制清零。这意味着 padding 位置本身仍会产生 attention 输出。

这不代表 padding 位置会参与有效 token 的信息聚合：

- 有效 query 无法读取 padding key。
- causal attention 下，较早的有效位置也无法读取右侧 padding。
- 训练损失会忽略 padding 标签，或由 SFT / DPO 的输出 mask 排除无效位置。

因此 padding query 的输出虽然存在，但通常不会进入有效 loss。这个结论描述的是项目当前实现；如果未来需要严格保证所有 padding 隐藏状态为零，还要增加 query 侧 mask 或在子层输出后清零。

## 12. 用 Attention Weight 聚合 V

softmax 后的权重形状是：

```text
attention_weight : (B, H, T, T)
V                : (B, H, T, Dh)
```

每个 query 位置对允许访问的 value 做加权求和：

```text
head_output[b, h, m, :]
  = sum_n attention_weight[b, h, m, n] * V[b, h, n, :]
```

代码对应的 einsum 是：

```text
bhmn,bhns->bhms
```

结果形状为：

```text
(B, H, T, Dh)
```

## 13. 合并 Head 与输出投影

各 head 的结果先恢复时间维在前的布局：

```text
(B, H, T, Dh)
  -> transpose / permute (B, T, H, Dh)
  -> reshape (B, T, D)
```

随后经过输出线性层：

```text
output = concat_heads W_o + b_o
```

输出形状仍为 `(B, T, D)`，因此可以在 `TransformBlock` 中与 attention 子层的原输入做残差相加。

## 14. LoRA 分支

当 `use_lora=True` 时，Q、K、V 和输出投影都会增加低秩分支：

```text
Linear_LoRA(x) = Linear_base(x) + scale * B(A(x))
scale = alpha / lora_rank
```

四个 LoRA B 层都初始化为零，所以启用 LoRA 的初始时刻不会改变基础投影的输出。

Attention 中的 LoRA 数据流为：

```text
x -> Q base + Q LoRA
x -> K base + K LoRA
x -> V base + V LoRA

attention 计算

concat_heads -> output base + output LoRA
```

`merge_lora_weights_inplace()` 可以把低秩增量合并到对应基础权重。Keras 与 PyTorch 的权重矩阵存储方向不同，所以矩阵乘法的代码顺序不同，但表达的是同一个增量 `B @ A`。

## 15. Keras 与 PyTorch 的代码对应关系

| 步骤 | Keras / TensorFlow | PyTorch |
| --- | --- | --- |
| 线性投影 | `Dense` | `nn.Linear` |
| 拆分 head | `tf.reshape` | `reshape` |
| 调整维度 | `tf.transpose` | `permute` |
| QK 点积 | `tf.einsum` | `torch.einsum` |
| softmax | `tf.nn.softmax` | `F.softmax(..., dim=-1)` |
| 合并 mask | `K.maximum` | `torch.maximum` |
| 前向入口 | `call()` | `forward()` |

除了框架 API 和 Keras / PyTorch 的权重布局差异，普通 `AttentionWithRoPE` 的计算顺序、mask 语义和张量形状一致。

## 16. 一次完整的 Attention 前向流程

```text
输入 x: (B, T, D)

1. Q = q_dense(x), K = k_dense(x), V = v_dense(x)
2. 如启用 LoRA，分别加入 Q/K/V 低秩增量
3. Q/K/V 拆分为 (B, H, T, Dh)
4. 对 Q 和 K 应用 RoPE
5. 计算 QK^T，得到 (B, H, T, T)
6. 构造 causal mask
7. 如存在 padding mask，与 causal mask 合并
8. 对屏蔽位置加入大负数
9. 除以 sqrt(Dh) 并在 key 维执行 softmax
10. attention weight 与 V 加权求和
11. 合并多个 head，恢复 (B, T, D)
12. 执行输出投影，如启用 LoRA则加入输出低秩增量
13. 返回 attention 输出，由 Transformer Block 完成残差相加
```

## 17. 普通 Attention 与 KVCache Attention 的边界

普通 `AttentionWithRoPE` 每次接收完整序列，并重新计算整段 Q、K、V。这适合训练，也可用于不带缓存的推理。

KVCache 推理仍使用相同的核心公式，但分为：

- Prefill：处理完整 prompt，并返回每层初始 K/V。
- Decode：每次只计算当前 token 的 Q/K/V，将新 K/V 写入 cache，再让当前 Q 读取有效 cache。

两套框架的普通 Attention 可以完全共用本文；KVCache 的外部形状和逐层组织存在差异，应分别阅读和维护对应实现说明。
