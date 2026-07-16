# 推理流程说明

本文记录项目从 prompt 到生成结果的完整推理链路。Keras / TensorFlow 与 PyTorch 使用相同的外层生成状态机、tokenizer、Top-K Sampling 和停止条件；模型调用、tensor 类型、设备管理及 KVCache 维度顺序按框架分别实现。

| 模块 | Keras / TensorFlow | PyTorch |
| --- | --- | --- |
| 推理控制 | `keras-mini-llm/interface.py` | `pytorch-mini-llm/interface.py` |
| Prefill / Decode | `keras-mini-llm/inference_models.py` | `pytorch-mini-llm/inference_models.py` |
| 采样 | `keras-mini-llm/sample_utils.py` | `pytorch-mini-llm/sample_utils.py` |
| Tokenizer | `keras-mini-llm/tokenizer.py` | `pytorch-mini-llm/tokenizer.py` |

两套 `sample_utils.py` 和 `tokenizer.py` 当前逻辑一致。KVCache 内部细节另见 [KVCache 原理与实现](kvcache.md)。

本文只讨论本项目当前的基础推理流程：

```text
prompt -> tokenizer -> prefill -> KVCache -> decode loop -> sampling -> stop
```

## 1. 整体流程

本项目的推理入口是：

```python
Interface.predict(prompts)
```

其中 `prompts` 是一个字符串列表：

```text
[
  prompt_0,
  prompt_1,
  ...
]
```

整体流程可以概括为：

```text
1. 对每个 prompt 做 BBPE encode
2. 过滤空 prompt 或达到 max_len 的 prompt
3. padding 成 batch
4. 调用 Prefill_Model 得到：
   - 下一 token logits
   - 初始 kcache
   - 初始 vcache
5. 把 kcache / vcache padding 到固定 max_len
6. 对 prefill logits 做 top-k sampling，得到第一个生成 token
7. 进入 decode loop：
   - 每轮输入上一轮生成的 token
   - 更新 kcache / vcache
   - 输出下一 token logits
   - top-k sampling 得到新 token
   - 删除已经结束的样本
8. 所有样本结束后返回结果
```

## 2. 初始化 Interface

Keras `Interface` 初始化时接收一个 tokenizer：

```python
interface = Interface(Tokenizer())
```

内部保存：

```python
self.tokenizer = tokenizer
self.vocab_size = len(self.tokenizer.vocab)
self.max_len = 1000
```

PyTorch `Interface` 还接收 `configs`，并保存 `num_block`、`num_head`、`embedding_dim`、权重路径和 `device`；Keras 则在初始化 Prefill/Decode 时分别传入 configs。

其中：

```text
tokenizer : 负责 encode / decode
vocab_size : 模型输出 logits 的词表大小
max_len : 本项目推理阶段允许的最大序列长度
```

## 3. 初始化 Prefill 和 Decode 模型

推理前需要分别初始化 Prefill 和 Decode。Keras 调用为：

```python
interface.init_prefill_model(configs)
interface.init_decode_model(configs)
```

PyTorch 在构造 `Interface(tokenizer, configs)` 时已保存配置，因此调用不再重复传参：

```python
interface.init_prefill_model()
interface.init_decode_model()
```

两个初始化函数都会从 `configs` 中读取：

```text
num_block
num_head
模型维度：Keras 为 embedding_size，PyTorch 为 embedding_dim
use_lora
weight_map_path
lora_weights_path（可选）
```

并从 tokenizer 中读取：

```python
special_ids = self.tokenizer.special_ids
```

### 3.1 Prefill_Model

`Prefill_Model` 处理完整 prompt：

```text
inputs: (batch, prompt_time)
```

Keras 输出 cache 为 batch-first：

```text
logits : (batch, prompt_time, vocab_size)
kcache : (batch, layer, prompt_time, head, head_dim)
vcache : (batch, layer, prompt_time, head, head_dim)
```

PyTorch logits 形状相同，但 cache 为：

```text
kcache/vcache: (layer, batch, head, prompt_time, head_dim)
```

### 3.2 Decode_Model

`Decode_Model` 每次处理一个 token：

```text
inputs: (batch, 1)
```

额外输入的逻辑内容相同，但 cache 形状不同：

```text
cur_valid_len : (batch,)
kcache/vcache:
  Keras   (batch, layer, max_len, head, head_dim)
  PyTorch (layer, batch, head, max_len, head_dim)
```

Keras 内部模型显式输出 logits 和更新后的 cache。PyTorch 内部 `nn.Module` 只返回 logits，cache 由原地更新保留；两套外层 `predict()` 最终都返回：

```text
logits : (batch, 1, vocab_size)
更新后的 kcache
更新后的 vcache
```

两个模型加载的是同一份训练权重映射；当 `use_lora=True` 且提供 `lora_weights_path` 时，还会在 base 权重之上加载 LoRA 增量：

```python
apply_train_weights(model, weight_map_path)
apply_lora_weights(model, lora_weights_path)
```

两个 `interface.py` 的示例配置都设置 `use_lora=False`，直接加载 DPO merged 权重：Keras 使用 `.pkl`，PyTorch 使用 `.pt`，因此无需再提供 LoRA 增量文件。

## 4. Prompt 编码与过滤

`Interface.predict()` 首先遍历所有 prompt：

```python
text_ids = self.tokenizer.encode_text(prompt)
text_ids = text_ids[:self.max_len]
```

每个样本在返回结果中对应一个字典：

```python
ret[i] = {
    "prompt": text_ids,
    "generated": [],
    "count": len(text_ids),
    "isover": False,
}
```

其中：

```text
prompt    : prompt 的 token ids
generated : 后续生成的 token ids
count     : 当前样本总 token 数
isover    : 当前样本是否已经结束
```

如果 prompt 为空，或者长度已经达到 `max_len`：

```python
if len(text_ids) == self.max_len or len(text_ids) == 0:
    ret[i]["isover"] = True
    continue
```

这类样本不会进入后续模型推理。

剩余样本会被加入：

```python
valid_prompt_ids
batch_text_ids
cur_valid_len
```

其中：

```text
valid_prompt_ids : 当前仍需要推理的原始样本编号
batch_text_ids   : 当前 batch 的 prompt token ids
cur_valid_len    : 每条样本真实 prompt 长度
```

## 5. Padding

由于 batch 内 prompt 长度不同，需要 padding 到当前 batch 的最大 prompt 长度：

```python
batch_text_ids = self.padding(batch_text_ids)
```

`padding()` 的逻辑是：

```python
max_len = max(len(x) for x in X)
X = [x + [pad_id] * (max_len - len(x)) for x in X]
```

Keras 返回 NumPy `int32` 数组；PyTorch 随后创建位于配置设备上的 `torch.long` tensor。

输出：

```text
batch_text_ids: (batch, prompt_time)
```

这里的 `prompt_time` 是当前 batch 的最大 prompt 长度，不一定等于全局 `self.max_len`。

## 6. Prefill 阶段

调用：

```python
preds, kcache, vcache = self.prefill_model.predict(batch_text_ids, cur_valid_len)
```

### 6.1 Prefill_Model 内部做什么

`Prefill_Model` 会：

```text
1. 根据 inputs 构造 padding mask
2. 计算 embedding
3. 逐层执行 TransformBlock_Prefill_KVCache
4. 得到 logits
5. 收集每层 kcache / vcache
```

模型输出的 `preds` 是完整 prompt 每个位置的 logits：

```text
preds: (batch, prompt_time, vocab_size)
```

### 6.2 取最后一个真实 prompt 位置的 logits

由于 prompt 做过 padding，不能直接取最后一列。

`Prefill_Model.predict()` 会根据：

```python
t_indices = cur_valid_len - 1
```

取每条样本最后一个真实 token 位置的 logits：

```python
preds = preds[batch_indices, t_indices]
```

得到：

```text
preds: (batch, vocab_size)
```

这个位置的 logits 用来预测 prompt 后面的第一个新 token。

### 6.3 禁止部分特殊 token

prefill 和 decode 的 `predict()` 中都会把这些 token 的 logits 设为很小：

```python
preds[:, special_ids["<bos>"]] = -1e10
preds[:, special_ids["<unk>"]] = -1e10
preds[:, special_ids["<pad>"]] = -1e10
```

也就是说：

```text
<bos> 不参与生成
<unk> 不参与生成
<pad> 不参与生成
```

注意：

```text
<eos> 没有被禁止，因为需要允许模型生成结束符。
```

Keras `model.predict()` 直接返回 NumPy 数据。PyTorch 的 Prefill/Decode 在 `torch.inference_mode()` 中运行，cache 保留在配置设备上，只有用于采样的 logits 会转到 CPU NumPy；因此共用的 `top_k_sampling()` 不需要区分框架。

## 7. 初始化固定长度 KVCache

Keras prefill 输出的 cache 形状是：

```text
(batch, layer, prompt_time, head, head_dim)
```

而 decode 模型需要固定长度：

```text
(batch, layer, max_len, head, head_dim)
```

因此 `interface.py` 会对 time 维做 padding：

```python
kcache = np.pad(
    kcache,
    pad_width=((0,0), (0,0), (0,self.max_len-kcache.shape[2]), (0,0), (0,0)),
    mode="constant",
    constant_values=0
)
```

`vcache` 同理。

这样后续 decode 可以在固定形状 cache 中按位置更新。

PyTorch prefill cache 为：

```text
(layer, batch, head, prompt_time, head_dim)
```

并使用：

```python
kcache = F.pad(kcache, (0, 0, 0, self.max_len - kcache.size(3)))
```

把第 3 维补成 `max_len`，得到 `(layer, batch, head, max_len, head_dim)`。两套实现的逻辑容量相同，只有维度顺序和 padding API 不同。

## 8. 第一次采样

prefill 得到的 `preds` 先经过：

```python
pred_ids = top_k_sampling(preds)
```

得到每条样本的第一个生成 token。

然后写入结果：

```python
ret[prompt_id]["generated"].append(pred_id)
ret[prompt_id]["count"] += 1
```

如果达到停止条件：

```python
ret[prompt_id]["count"] == self.max_len
```

或者生成了：

```python
pred_id == self.tokenizer.special_ids["<eos>"]
```

则标记：

```python
ret[prompt_id]["isover"] = True
```

否则该样本进入后续 decode loop。

需要注意：

```text
prefill 后采样得到的第一个生成 token，此时还没有写入 KVCache。
它会在下一轮 decode 中作为输入 token，再写入 cache。
```

## 9. Decode Loop

如果还有未结束样本，就进入：

```python
while True:
```

每一轮 decode 会构造新的 batch。

### 9.1 当前输入 token

对每个仍在生成的样本，当前 decode 输入是上一轮刚生成的 token：

```python
text_ids = [ret[prompt_id]["generated"][-1]]
```

因此：

```text
batch_text_ids: (batch, 1)
```

### 9.2 当前 cur_valid_len

同时计算：

```python
cur_valid_len.append(len(ret[prompt_id]["prompt"] + ret[prompt_id]["generated"]))
```

也就是说，decode 阶段的 `cur_valid_len` 包含当前输入 token。

例如：

```text
prompt 长度 = L
已经生成 token_1
当前 decode 输入 token_1
cur_valid_len = L + 1
```

因此当前 token 写入 cache 的位置是：

```text
cur_valid_len - 1 = L
```

这正好接在 prompt 后面。

### 9.3 调用 Decode_Model

两套外层接口的调用形式一致：

```python
preds, kcache, vcache = self.decode_model.predict(
    batch_text_ids,
    cur_valid_len,
    kcache,
    vcache
)
```

`Decode_Model` 内部会：

```text
1. 对当前 token 做 embedding
2. 逐层计算当前 token 的 q/k/v
3. 对当前 q/k 应用 RoPE
4. 把当前 k/v 写入 cache
5. 当前 q attend 到 cache 中所有有效 k/v
6. 输出当前 token 位置的 logits
```

由于 decode 输入长度固定为 1，`Decode_Model.predict()` 直接取：

```python
preds = preds[:, -1]
```

得到：

```text
preds: (batch, vocab_size)
```

这个 logits 用来预测下一个 token。

### 9.4 采样并更新状态

每轮 decode 后再次采样：

```python
pred_ids = top_k_sampling(preds)
```

并写入：

```python
ret[prompt_id]["generated"].append(pred_id)
ret[prompt_id]["count"] += 1
```

如果达到停止条件，则标记结束；否则继续下一轮。

## 10. Batch 中样本提前结束

batch 推理时，不同样本可能在不同时间结束。

本项目用：

```python
valid_prompt_ids
```

维护仍在生成的样本编号。

每轮采样后构造：

```python
new_valid_prompt_ids
```

如果某些样本已经结束，就从后续 batch 中移除。

Keras cache 是 batch-first：

```text
(batch, layer, max_len, head, head_dim)
```

所以可以直接筛选 batch 维：

```python
kcache = kcache[np.array(valid_prompt_ids_indices)]
vcache = vcache[np.array(valid_prompt_ids_indices)]
```

PyTorch cache 的 batch 在第 1 维，并使用设备上的 index tensor：

```python
kcache = kcache[:, keep_indices]
vcache = vcache[:, keep_indices]
```

如果所有样本都结束：

```python
return ret
```

## 11. Top-K Sampling

两套完全相同的采样函数都位于各自的 `sample_utils.py`：

```python
top_k_sampling(logits, k=70, temperature=1.0)
```

输入：

```text
logits: (batch, vocab_size)
```

流程是：

```text
1. logits / temperature
2. 取每条样本 top-k token
3. 对 top-k logits 做 softmax
4. 在 top-k 范围内按概率采样
```

其中：

```python
top_k_indices = np.argpartition(-logits, k-1, axis=-1)[...,:k]
top_k_logits = np.take_along_axis(logits, top_k_indices, axis=-1)
```

然后：

```python
chosen_id = np.random.choice(indices, p=probs)
```

最终返回：

```text
chosen_ids
```

也就是 batch 中每条样本采样到的 token id。

## 12. 结果返回与 decode

`Interface.predict()` 返回的是 `ret` 字典。

每个样本包含：

```text
prompt    : prompt token ids
generated : generated token ids
count     : token 总数
isover    : 是否结束
```

如果需要得到文本，可以把 prompt 和 generated 拼起来，再调用 tokenizer decode：

```python
text = tokenizer.decode(ret[i]["prompt"] + ret[i]["generated"])
```

## 13. 关键约定总结

本项目推理流程中有几个关键约定：

- `Prefill_Model` 输入完整 prompt，输出第一个新 token 的 logits 和初始 KVCache。
- `Decode_Model` 每次输入上一轮生成的一个 token。
- prefill 后第一次采样得到的 token，会在下一轮 decode 中写入 cache。
- `cur_valid_len` 在 decode 中包含当前输入 token。
- decode 当前 token 的 cache 写入位置是 `cur_valid_len - 1`。
- 生成时禁止 `<bos>`、`<unk>`、`<pad>`，但允许 `<eos>`。
- 生成 `<eos>` 或达到 `max_len` 后，样本停止生成。
- batch 中提前结束的样本会被移除；Keras 沿 cache 第 0 维裁剪，PyTorch 沿第 1 维裁剪。

整体链路可以浓缩为：

```text
encode prompts
-> pad batch
-> prefill once
-> pad cache to max_len
-> sample first token
-> decode one token at a time
-> sample next token
-> prune finished samples
-> return token ids
```
