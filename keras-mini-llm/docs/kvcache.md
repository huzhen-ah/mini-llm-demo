# KVCache 原理与实现

本文记录本项目中 KVCache 的设计思路和实现细节，对应代码主要在：

- `inference_models.py`
- `attention.py`
- `transformblock.py`
- `interface.py`

本文只讨论本项目里的基础 KVCache：prefill 一次性生成 prompt 的初始 k/v cache，decode 阶段每次更新一个新 token 的 k/v。

## 1. 为什么需要 KVCache

自回归语言模型每次生成一个新 token：

```text
prompt -> token_1 -> token_2 -> token_3 -> ...
```

如果不使用 KVCache，第 `t` 次生成时，需要把前面所有 token 再完整送进模型，重新计算每一层的 key / value：

```text
step 1: 计算 prompt 的 k/v
step 2: 重新计算 prompt + token_1 的 k/v
step 3: 重新计算 prompt + token_1 + token_2 的 k/v
...
```

这会造成大量重复计算。

KVCache 的核心想法是：

```text
历史 token 的 k/v 一旦算出来，就保存下来；
后续 decode 只计算当前新 token 的 k/v，然后追加到 cache 中。
```

这样每一步只需要计算当前 token 的 q/k/v。其中只有当前 token 产生 query，这个 query 会和 cache 中所有有效 key/value 做 attention。k/v 则来自：

```text
历史 cache + 当前 token 的 k/v
```

## 2. Prefill 和 Decode

使用 KVCache 时，推理通常分成两个阶段：

```text
prefill 阶段：一次性处理完整 prompt，生成初始 kcache / vcache
decode 阶段：每次只输入一个新 token，更新 kcache / vcache
```

本项目中对应两个模型：

```text
Prefill_Model
Decode_Model
```

拆成两个模型的原因是：

- prefill 的输入是完整 prompt，长度可以大于 1。
- decode 的输入是单个 token，长度固定为 1。
- decode 需要额外输入 `cur_valid_len`、`kcache`、`vcache`。
- Keras Functional Model 对输入输出结构比较敏感，拆开后控制逻辑更清晰。

这两个模型使用同一套训练权重，通过 `weight_utils.py` 中的权重路径映射加载参数。

## 3. Cache 的形状设计

### 3.1 对外形状

在 decode 输入输出侧，固定长度 cache 使用 batch-first 形状：

```text
(batch, layer, max_time, head, head_dim)
```

含义是：

```text
batch    : 当前仍在生成的样本数量
layer    : Transformer block 数量
max_time : 最大生成长度 / cache 最大长度
head     : attention head 数量
head_dim : 每个 head 的维度
```

这个形状的好处是：当 batch 中某些样本提前生成 `<eos>` 时，可以直接在第 0 维筛选仍然需要继续生成的样本：

```python
kcache = kcache[keep_indices]
vcache = vcache[keep_indices]
```

### 3.2 Decode 模型内部形状

`Decode_Model` 接收 batch-first cache 后，会先转成 layer-first：

```python
kcache = K.transpose(kcache_input, axes=[1, 2, 0, 3, 4])
vcache = K.transpose(vcache_input, axes=[1, 2, 0, 3, 4])
```

内部形状变成：

```text
(layer, max_time, batch, head, head_dim)
```

这样每个 Transformer block 可以通过 `self.cur_layer` 取出当前层的 cache：

```python
kcache[self.cur_layer]
vcache[self.cur_layer]
```

当前层取出后形状是：

```text
(max_time, batch, head, head_dim)
```

再转成 attention 需要的格式：

```python
k = K.transpose(kcache[self.cur_layer], axes=[1, 2, 0, 3])
v = K.transpose(vcache[self.cur_layer], axes=[1, 2, 0, 3])
```

得到：

```text
(batch, head, max_time, head_dim)
```

### 3.3 为什么 decode 输出还要转回 batch-first

`Decode_Model` 内部计算结束后，输出前会把 cache 转回：

```python
out_kcache = Lambda(lambda x : K.transpose(x, axes=[2,0,1,3,4]))(...)
out_vcache = Lambda(lambda x : K.transpose(x, axes=[2,0,1,3,4]))(...)
```

也就是：

```text
(layer, max_time, batch, head, head_dim)
-> (batch, layer, max_time, head, head_dim)
```

这样下一轮 decode 可以继续以 batch-first 的形式传入，也方便 `interface.py` 按 batch 维删除已经结束的样本。

## 4. cur_valid_len 的含义

本项目中 `cur_valid_len` 表示：

```text
每条样本当前真实有效 token 数量，不包括 padding。
```

在 decode 阶段有一个更具体的约定：

```text
cur_valid_len 表示“当前 token 写入 cache 后”的有效长度。
```

因此当前 token 应该写入的位置是：

```text
cur_valid_len - 1
```

例如 prompt 长度为 5，prefill 后 cache 中有效位置是：

```text
0, 1, 2, 3, 4
```

prefill 输出第一个生成 token 后，下一次 decode 会把这个 token 作为输入。此时：

```text
cur_valid_len = 6
```

所以这个 token 写入 cache 的位置是：

```text
cur_valid_len - 1 = 5
```

也就是正好接在 prompt 后面。

这个约定同时用于：

- RoPE 的 position。
- KVCache 的写入位置。
- decode 阶段的有效 key mask。

三者必须保持一致。

## 5. Prefill 阶段

### 5.1 输入和 padding mask

`Prefill_Model` 的输入是完整 prompt：

```python
inputs = Input(shape=(None,), dtype="int32", name="inputs")
```

batch 内不同 prompt 长度不一样，因此 `interface.py` 会先 padding 到当前 batch 的最大 prompt 长度：

```python
batch_text_ids = self.padding(batch_text_ids)
```

然后在模型内部构造 padding mask：

```python
padding_mask = K.cast(K.equal(inputs, self.special_ids["<pad>"]), dtype="float32")
```

形状是：

```text
(batch, prompt_time)
```

其中：

```text
1 表示 pad token
0 表示真实 token
```

### 5.2 prefill attention

在 `AttentionWithRoPE_Prefill_KVCache` 中，q/k/v 先被投影并 reshape：

```text
q, k, v: (batch, head, prompt_time, head_dim)
```

然后对 q/k 应用 RoPE：

```python
q = rope_exp(q)
k = rope_exp(k)
```

prefill 阶段会使用两类 mask：

```text
causal mask  : 防止当前位置看见未来 token
padding mask : 防止真实 token 看见 pad token
```

两者合并后加到 attention score 上。

### 5.3 prefill 输出 cache

prefill 阶段计算 padding 后 prompt 的 k/v。当前层返回：

```python
return out, K.transpose(k, axes=[2,0,1,3]), K.transpose(v, axes=[2,0,1,3])
```

也就是：

```text
k/v: (batch, head, prompt_time, head_dim)
->   (prompt_time, batch, head, head_dim)
```

`Prefill_Model` 会收集每一层的 cache：

```python
kcache.append(cur_layer_kcache)
vcache.append(cur_layer_vcache)
```

再 stack：

```python
kcache = K.stack(kcache, axis=0)
vcache = K.stack(vcache, axis=0)
```

此时形状是：

```text
(layer, prompt_time, batch, head, head_dim)
```

然后转成对外 batch-first：

```python
kcache = K.transpose(kcache, axes=[2,0,1,3,4])
vcache = K.transpose(vcache, axes=[2,0,1,3,4])
```

得到：

```text
(batch, layer, prompt_time, head, head_dim)
```

### 5.4 padding 到 max_len

prefill 输出的 `prompt_time` 只是当前 batch padding 后的 prompt 长度，不是固定最大长度。

为了让 decode 模型接收固定形状 cache，`interface.py` 会把 time 维 padding 到 `max_len`：

```python
kcache = np.pad(
    kcache,
    pad_width=((0,0), (0,0), (0,self.max_len-kcache.shape[2]), (0,0), (0,0)),
    mode="constant",
    constant_values=0
)
```

vcache 同理。

最终传给 decode 的 cache 形状是：

```text
(batch, layer, max_len, head, head_dim)
```

## 6. Decode 阶段

### 6.1 decode 输入

`Decode_Model` 每次只输入一个 token：

```python
inputs = Input(shape=(1,), dtype="int32", name="inputs")
```

同时输入：

```text
cur_valid_len
kcache
vcache
```

其中：

```text
cur_valid_len: (batch,)
kcache/vcache: (batch, layer, max_len, head, head_dim)
```

### 6.2 当前 token 的 q/k/v

decode 阶段当前 token 经过 dense 后：

```text
q, k, v: (batch, 1, embedding_size)
```

reshape 后：

```text
q, k, v: (batch, head, 1, head_dim)
```

由于 decode 输入只有当前 token，因此 RoPE position 使用：

```text
cur_valid_len - 1
```

代码中：

```python
q = rope_exp(q, cur_valid_len=cur_valid_len)
k = rope_exp(k, cur_valid_len=cur_valid_len)
```

### 6.3 更新 cache

当前 token 的 k/v 需要写入 cache。

内部 cache 形状是：

```text
(layer, max_time, batch, head, head_dim)
```

为了用 `tf.tensor_scatter_nd_update` 按 `[layer, batch, time]` 写入，代码先转置：

```python
cache = K.transpose(cache, axes=[0,2,1,3,4])
```

得到：

```text
(layer, batch, max_time, head, head_dim)
```

然后构造 indices：

```python
batch_indices = K.reshape(K.arange(b, dtype="int32"), (-1,1))
t_indices = K.reshape(cur_valid_len - 1, (-1,1))
layer_indices = K.ones(shape=(b,1), dtype="int32") * self.cur_layer
indices = K.concatenate([layer_indices, batch_indices, t_indices], axis=-1)
```

`indices` 形状是：

```text
(batch, 3)
```

每一行表示：

```text
[layer_index, batch_index, time_index]
```

当前 token 的 k/v 从：

```text
(batch, head, 1, head_dim)
```

squeeze 成：

```text
(batch, head, head_dim)
```

然后 scatter update：

```python
cache = tf.tensor_scatter_nd_update(cache, indices, cur_layer_k_or_v_cache)
```

更新完成后再转回：

```python
cache = K.transpose(cache, axes=[0,2,1,3,4])
```

回到：

```text
(layer, max_time, batch, head, head_dim)
```

### 6.4 decode attention 使用完整 cache

当前层 cache 取出后：

```python
k = K.transpose(kcache[self.cur_layer], axes=[1,2,0,3])
v = K.transpose(vcache[self.cur_layer], axes=[1,2,0,3])
```

得到：

```text
k/v: (batch, head, max_time, head_dim)
```

当前 q 的形状是：

```text
q: (batch, head, 1, head_dim)
```

attention score：

```python
qk = tf.einsum("bhms,bhns->bhmn", q, k)
```

形状是：

```text
(batch, head, 1, max_time)
```

### 6.5 decode 阶段不需要 causal mask

prefill 阶段 q/k 都是完整序列，因此需要 causal mask。

decode 阶段 q 只有当前 token：

```text
query_len = 1
```

而 cache 中有效 key 的范围由 `cur_valid_len` 控制。

因此 decode 阶段不再使用 causal mask，只需要屏蔽无效 key：

```python
cur_valid_mask = cur_valid_len[:, None]
mask = K.cast(cur_valid_mask > K.arange(K.shape(kcache)[1], dtype="int32"), "float32")
mask = 1 - mask
```

这里：

```text
position < cur_valid_len  -> valid
position >= cur_valid_len -> invalid
```

mask 最后变成：

```text
(batch, 1, 1, max_time)
```

再乘上 `-1e10` 加到 attention score 上。

这会屏蔽：

- prompt padding 位置。
- 还没有生成到的未来 cache 位置。
- max_len 中的空白位置。

## 7. Batch 中样本提前结束

batch 推理时，有些样本会先生成 `<eos>`，或者达到 `max_len`。

`interface.py` 中会维护：

```python
valid_prompt_ids
```

每轮采样后，只保留尚未结束的样本：

```python
new_valid_prompt_ids = []
```

如果 batch 变小，就按 batch 维筛选 cache：

```python
kcache = kcache[np.array(valid_prompt_ids_indices)]
vcache = vcache[np.array(valid_prompt_ids_indices)]
```

由于对外 cache 是：

```text
(batch, layer, max_time, head, head_dim)
```

所以删除完成样本很直接。

这也是本项目把对外 cache 设计成 batch-first 的原因之一。

## 8. 为什么 cache 中可以有 padding

batch 内 prompt 长度不同，prefill 会 padding 到当前 batch 的最大 prompt 长度。

因此对某个短 prompt 来说，prefill 后 cache 中可能是：

```text
真实 prompt + prompt padding + 后续 max_len padding
```

但真正有效的部分由 `cur_valid_len` 控制：

```text
position < cur_valid_len
```

decode 时，新生成 token 会写入：

```text
cur_valid_len - 1
```

也就是覆盖短 prompt 后面的第一个无效位置。

随着生成继续，新的 token 会依次覆盖后面的 padding 位置。

所以 cache 的物理形状固定，但每条样本的逻辑有效长度由 `cur_valid_len` 决定。

## 9. 当前实现的约定和注意点

本项目当前 KVCache 实现有几个重要约定：

- decode 对外 cache 使用 `(batch, layer, max_time, head, head_dim)`。
- Decode 模型内部 cache 使用 `(layer, max_time, batch, head, head_dim)`。
- scatter update 临时使用 `(layer, batch, max_time, head, head_dim)`。
- `cur_valid_len` 在 decode 阶段表示当前 token 写入后的有效长度。
- 当前 token 写入位置是 `cur_valid_len - 1`。
- k 写入 cache 前已经应用 RoPE，q 参与 attention 前已经应用 RoPE。
- v 写入 cache 前不应用 RoPE。
- decode 阶段只输入一个 token，因此不需要 causal mask。
- decode 阶段通过 `cur_valid_len` 构造 key mask，屏蔽所有无效 cache 位置。

这些约定必须整体保持一致。KVCache 最容易出错的地方不是单个公式，而是：

```text
cache shape、cur_valid_len、RoPE position、mask 范围、scatter 写入位置
```

这几件事只要有一个约定不一致，就会出现很隐蔽的推理错误。

## 10. 总结

本项目的 KVCache 流程可以概括为：

```text
Prefill:
完整 prompt -> 每层计算 q/k/v -> 保存 RoPE 后的 k 和未旋转的 v -> 输出初始 cache

Decode:
单个 token -> 计算当前 q/k/v -> 当前 k/v 写入 cache -> 当前 q attend 到有效 cache
```

核心设计是：

```text
固定形状 cache + cur_valid_len 控制有效范围 + tensor_scatter_nd_update 按 batch 更新单 token 位置。
```

这样可以在保持矩阵计算形式的同时，避免每一步重复计算历史 token 的 k/v。
