# LoRA-DPO 原理与实现

本文记录本项目中 LoRA-DPO 的设计思路和实现细节，对应代码主要在：

- `lora_dpo.py`
- `train_utils.py`
- `losses.py`
- `lora_utils.py`
- `merge_lora_checkpoint.py`

本文只讨论本项目里的基础 DPO：在已经完成 SFT 的模型上继续训练 LoRA，使模型更偏向 chosen answer，远离 rejected answer。

## 1. DPO 要解决什么问题

SFT 学的是：

```text
prompt -> answer
```

也就是让模型模仿一条标准答案。

DPO 学的是：

```text
同一个 prompt 下，chosen answer 应该比 rejected answer 更好。
```

因此 DPO 数据不是单条回答，而是一组成对偏好：

```json
{
  "prompt": "乔峰的悲剧主要来自哪里？",
  "chosen": "乔峰的悲剧来自身世、民族身份和江湖责任之间的冲突。",
  "rejected": "乔峰的悲剧主要是因为他武功不够高。"
}
```

本项目的目标不是做完整 RLHF，而是用 DPO loss 直接训练模型，让模型提高 chosen 序列相对 rejected 序列的概率。

## 2. 当前训练链路

本项目中 DPO 的链路是：

```text
pretrain base
  -> LoRA-SFT
  -> merge SFT LoRA into base
  -> load SFT merged base
  -> precompute ref logp
  -> LoRA-DPO
  -> save DPO LoRA
```

对应入口：

```text
merge_lora_checkpoint.py
lora_dpo.py
```

其中 `merge_lora_checkpoint.py` 用于把 SFT LoRA 合并进 base 权重：

```text
base weight + SFT LoRA delta -> SFT merged base
```

然后 `lora_dpo.py` 在这个 SFT merged base 上继续训练新的 DPO LoRA。

## 3. 为什么要先算 ref logp

DPO loss 里有两个模型：

```text
policy model : 当前正在训练的模型
reference model : 参考模型
```

在本项目里，DPO 开始前的 SFT merged model 就是 reference model。训练 DPO 时，reference model 不更新。

因此本项目先调用：

```python
X_train = pre_infer_dpo_data(X_train, model, eos_id, pad_id)
```

提前计算每条 chosen/rejected 序列中每个 token 的 reference log probability，并保存回样本中：

```python
X[index]["chosen_logp"] = chosen_logp
X[index]["rejected_logp"] = rejected_logp
```

这样训练时不需要再跑一遍 reference model，DPO loss 只需要读取已经算好的 ref logp。

## 4. Sequence Log Probability

语言模型不是一次性给整句话一个概率，而是逐 token 预测。

例如 answer 有三个 token：

```text
a0, a1, a2
```

那么序列概率是：

```text
p(a0 | prompt)
* p(a1 | prompt, a0)
* p(a2 | prompt, a0, a1)
```

取 log 后变成加法：

```text
log p(a0 | prompt)
+ log p(a1 | prompt, a0)
+ log p(a2 | prompt, a0, a1)
```

所以 DPO 中比较 chosen 和 rejected，本质比较的是两个 answer 的 token logp 之和。

## 5. 数据加载

`load_dpo_data()` 读取每行 json：

```json
{"prompt": "...", "chosen": "...", "rejected": "..."}
```

然后分别编码：

```python
prompt_ids = tokenizer_tool.encode_text(prompt)
chosen_ids = tokenizer_tool.encode_text(chosen)
rejected_ids = tokenizer_tool.encode_text(rejected)
```

拼出两条序列：

```python
prompt_chosen_ids = prompt_ids + chosen_ids
prompt_rejected_ids = prompt_ids + rejected_ids
```

并构造 answer-only mask：

```python
prompt_chosen_mask = [0] * (len(prompt_ids) - 1) + [1] * len(chosen_ids) + [1]
prompt_rejected_mask = [0] * (len(prompt_ids) - 1) + [1] * len(rejected_ids) + [1]
```

这里最后一个 `1` 对应 `<eos>`。因为训练时会做 next-token shift，所以 mask 长度与 label 侧对齐。

## 6. ref logp 预计算

`pre_infer_dpo_data()` 会临时把 chosen/rejected 拼到同一个 batch 中：

```text
batch_inputs : chosen + rejected
batch_outputs: chosen_label + rejected_label
```

模型输出 logits 后，先做 softmax，再取目标 token 对应概率：

```python
batch_logp = np.log(softmax(batch_preds))
batch_logp = np.take_along_axis(batch_logp, batch_outputs[..., None], axis=-1)[:, :, 0]
```

这里得到的是：

```text
每个位置上，reference model 对真实 target token 的 log probability。
```

之后保存到样本里，供训练阶段的 DPO loss 使用。

## 7. DPO 训练 batch

`data_generator_dpo()` 每次取一批 preference pairs，然后拆成两批序列：

```text
chosen batch
rejected batch
```

最后在 batch 维拼起来：

```text
batch_input.shape  = (batch * 2, m)
batch_output.shape = (batch * 2, m, 3)
```

其中前 `batch` 条是 chosen，后 `batch` 条是 rejected。

`batch_output` 最后一维含义是：

```text
[..., 0] : target token id
[..., 1] : reference logp
[..., 2] : loss mask
```

也就是：

```text
y_true.shape = (batch * 2, m, 3)
y_pred.shape = (batch * 2, m, vocab_size)
```

## 8. 为什么 batch 要乘 2

因为一条 DPO 样本天然包含两条回答：

```text
prompt + chosen
prompt + rejected
```

为了只跑一次 policy model，本项目把二者拼在 batch 维：

```text
chosen   -> y_pred[:batch]
rejected -> y_pred[batch:]
```

然后在 loss 函数中再拆回来。

## 9. DPO Loss

`dpo_loss()` 先从 `y_true` 中拆出 chosen/rejected 的 token id、reference logp 和 mask：

```python
ref_chosen_ids = y_true[:batch, :, 0]
ref_chosen_logp = y_true[:batch, :, 1]
ref_chosen_mask = y_true[:batch, :, 2]
```

policy model 的 logp 来自当前 `y_pred`：

```python
chosen_logp = K.take_along_axis(
    K.log_softmax(chosen_pred, axis=-1),
    ref_chosen_ids[:, :, None],
    axis=-1
)[:, :, 0]
```

然后用 mask 只保留 answer/eos 部分：

```python
chosen_logp = chosen_logp * ref_chosen_mask
```

chosen 和 rejected 的 sequence logp 分别是：

```text
sum(chosen token logp)
sum(rejected token logp)
```

DPO 核心比较量是：

```text
(policy_chosen - policy_rejected)
- (ref_chosen - ref_rejected)
```

代码中写成：

```python
tmp = sum(policy_chosen)
    - sum(policy_rejected)
    - sum(ref_chosen)
    + sum(ref_rejected)
```

最后乘上 `beta`：

```python
tmp = beta * tmp
loss = -mean(log_sigmoid(tmp))
```

如果当前 policy 比 reference 更偏向 chosen，`tmp` 会变大，loss 会下降。

## 10. beta 的含义

`beta` 控制 DPO 约束强度：

```python
def dpo_loss(beta=0.1):
```

在本项目中默认使用：

```text
beta = 0.1
```

它只是一个常用的起点。demo 中重点不是调参，而是把 DPO 的数据结构、ref logp 和 pairwise loss 跑通。

## 11. DPO 与 LoRA

本项目中 DPO 仍然只训练 LoRA 参数：

```python
mark_only_lora_as_trainable(model)
```

也就是说：

```text
SFT merged base : 固定
DPO LoRA        : 训练
```

训练结束后，`DPO_Evaluate` 会保存 DPO LoRA 权重：

```text
lora_dpo_weights/{epoch}_lora_weights.pkl
```

如果后续需要部署，也可以复用 `merge_lora_checkpoint.py` 把 DPO LoRA 合并进 SFT merged base。但本项目当前默认保留 DPO LoRA checkpoint。

## 12. 关于动态 padding

`data_generator_dpo()` 当前使用 batch 内动态 padding：

```text
每个 batch pad 到当前 batch 的最大长度。
```

这样更省计算，也更符合手写训练逻辑。

需要注意的是，部分 Keras 版本在 `model.fit(generator)` 时会根据前几个 batch 推断固定 shape。如果不同 batch 的 time 维不同，可能触发 shape 推断错误。

这属于 Keras `fit(generator)` 的接口限制，不是 DPO 数据逻辑本身的问题。实际处理时可以选择：

```text
1. 固定 pad 到 context_size
2. 使用 tf.data.Dataset.from_generator 并显式声明 output_signature
3. 使用自定义训练循环
4. 在 PyTorch 版本中自然支持 batch 间动态长度
```

本项目保留动态 padding，因为它更直接地表达了数据对齐逻辑。

## 13. 小结

本项目的 DPO 实现核心是：

```text
precompute ref logp
chosen/rejected 拼成 batch*2
y_true 保存 target_id/ref_logp/mask
loss 中重新 gather policy logp
按序列 logp 差计算 DPO loss
只训练 LoRA 参数
```

它不是工业级 DPO 训练框架，但能清楚展示 DPO 从 preference pair 到 loss 的完整张量链路。
