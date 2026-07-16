# LoRA-SFT 原理与实现

本文记录项目中 LoRA-SFT 的设计思路和实现细节。Keras / TensorFlow 与 PyTorch 使用相同的样本定义、answer-only loss、LoRA 接入位置和权重合并目标；数据迭代、训练循环及 checkpoint 格式按框架分别实现。

| 模块 | Keras / TensorFlow | PyTorch |
| --- | --- | --- |
| 训练入口 | `keras-mini-llm/lora_sft.py` | `pytorch-mini-llm/lora_sft.py` |
| 数据管线 | `keras-mini-llm/train_utils.py` | `pytorch-mini-llm/train_utils.py` |
| Loss / Accuracy | `keras-mini-llm/losses.py`、`metrics.py` | `pytorch-mini-llm/losses.py`、`metrics.py` |
| LoRA 管理 | `keras-mini-llm/lora_utils.py` | `pytorch-mini-llm/lora_utils.py` |
| LoRA 层 | `keras-mini-llm/attention.py`、`layers.py` | `pytorch-mini-llm/attention.py`、`layers.py` |
| 评估与保存 | `keras-mini-llm/callbacks.py` | `pytorch-mini-llm/callbacks.py` |

本文只讨论本项目里的基础 LoRA-SFT：在预训练模型基础上加载 base 权重，冻结非 LoRA 参数，只训练 LoRA 参数，并使用 answer-only loss mask 做 instruction tuning。

## 1. SFT 要解决什么问题

预训练阶段做的是 next-token prediction：

```text
给定前面的 token，预测下一个 token。
```

这会让模型学到语言模式，但不一定学会按指令回答问题。SFT 的目标是让模型看到类似：

```text
用户问题 -> 助手回答
```

这样的样本，并只在 assistant answer 部分计算 loss。也就是说，prompt 部分只是条件，不希望模型因为“预测 prompt 本身”而被训练。

本项目中 SFT 的训练链路是：

```text
messages jsonl
  -> load_sft_data
  -> 动态 padding 与错位标签
  -> answer-only loss / accuracy
  -> LoRA 参数更新
  -> 保存 LoRA checkpoint
  -> 合并并保存新 base checkpoint
```

## 2. 数据格式

当前 SFT 数据使用 messages 格式，每行一个 json：

```json
{"messages": [{"role": "user", "content": "华筝和郭靖是什么关系？"}, {"role": "assistant", "content": "华筝是成吉思汗之女，曾与郭靖有婚约。"}]}
```

`load_sft_data()` 会遍历每条样本中的 messages：

```text
role == "user"      -> instruction
role == "assistant" -> output
```

然后分别用 tokenizer 编码：

```python
content_ids = tokenizer_tool.encode_text(content)
```

得到的数据结构是：

```python
{
    "instruction": [...],
    "input": "",
    "output": [...]
}
```

其中 `input` 当前没有实际使用，保留它只是为了以后扩展成 instruction + input + output 形式。

## 3. 样本拼接

生成训练样本时，一条 SFT 样本会被拼成：

```python
x = instruction + output + [eos_id]
```

也就是：

```text
prompt tokens + answer tokens + eos
```

这里的 `<eos>` 是直接拼特殊 token id，而不是把字符串 `"<eos>"` 送进 BBPE tokenizer。原因是：特殊 token 是模型协议中的控制符号，不应该被当成普通文本字节再切分。

## 4. Answer-Only Loss Mask

SFT 训练时不希望 prompt 部分参与 loss，只希望 answer 和 eos 参与 loss。因此原始 mask 是：

```python
mask = [0] * len(instruction) + [1] * len(output) + [1]
```

含义是：

```text
instruction : 0，不算 loss
output      : 1，计算 loss
eos         : 1，计算 loss
```

例如：

```text
instruction = [p0, p1, p2]
output      = [a0, a1]
eos         = [eos]
```

拼接后：

```text
x    = [p0, p1, p2, a0, a1, eos]
mask = [ 0,  0,  0,  1,  1,   1]
```

语言模型训练时输入和标签要错一位：

```python
X = X[:, :-1]
Y = X[:, 1:]
Mask = Mask[:, 1:]
```

所以实际训练的是：

```text
input : [p0, p1, p2, a0, a1]
label : [p1, p2, a0, a1, eos]
mask  : [ 0,  0,  1,  1,   1]
```

这表示：

```text
p0 -> p1 不算 loss
p1 -> p2 不算 loss
p2 -> a0 算 loss
a0 -> a1 算 loss
a1 -> eos 算 loss
```

也就是说，模型从最后一个 prompt token 开始学习预测第一个 answer token。

## 5. y_true 的结构

普通 next-token prediction 的 `y_true` 只需要 token id：

```text
y_true: (batch, time)
```

SFT 还需要传入 loss mask，所以本项目把 label 和 mask 拼在最后一维：

```python
Y = np.concatenate([Y, Mask], axis=-1)
```

最终形状是：

```text
Y: (batch, time, 2)
```

其中：

```text
Y[..., 0] : label token id
Y[..., 1] : sft loss mask
```

这样就不需要额外给 loss 函数传参，`sft_loss()` 可以直接从 `y_true` 中拆出 label 和 mask。

## 6. SFT Loss

`sft_loss()` 的核心逻辑是：

```python
y_true, mask = y_true[..., 0], y_true[..., 1]
ce_loss = sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
masked_loss = ce_loss * mask
loss = sum(masked_loss) / sum(mask)
```

也就是：

```text
只统计 mask == 1 的位置。
```

prompt 部分仍然参与前向计算，因为 answer 要依赖 prompt；但 prompt label 不参与 loss。

## 7. SFT Accuracy

两套实现的 accuracy 和 loss 使用同一个 mask：

```python
y_true, mask = y_true[..., 0], y_true[..., 1]
pred_labels = argmax(y_pred, axis=-1)
correct = equal(pred_labels, y_true) & (mask > 0)
```

因此 accuracy 只反映 answer/eos 部分的预测准确率，不会被 prompt token 影响。Keras 将它实现为有状态的 `SFTAccuracy` metric；PyTorch 的 `sft_accuracy()` 返回当前 batch 的正确 token 数和有效 token 数，再由训练循环累计。

## 8. LoRA 接入位置

本项目当前在两个模块中接入 LoRA：

```text
Attention:
  q / k / v / out

SwiGLU:
  gate_v / gate_w / out
```

### 8.1 Attention LoRA

以 q projection 为例，base 路径是：

```python
q = self.q_dense(x)
```

LoRA 路径是：

```python
q = q + self.scale * self.lora_q_B(self.lora_q_A(x))
```

其中：

```text
lora_q_A: hidden_dim -> lora_rank
lora_q_B: lora_rank -> hidden_dim
scale   : alpha / lora_rank
```

q/k/v/out 都采用同样结构。

### 8.2 SwiGLU LoRA

SwiGLU 中有三条 base dense：

```text
v_dense
w_dense
out_dense
```

对应接入：

```text
lora_gate_v_A / lora_gate_v_B
lora_gate_w_A / lora_gate_w_B
lora_out_A    / lora_out_B
```

其中 gate_v 和 gate_w 分别作用在 SwiGLU 的两个分支上，out LoRA 作用在 FFN 输出投影上。

## 9. Zero Initialization

LoRA 的 B 矩阵使用 zero initialization：

```python
kernel_initializer="zeros"
```

这样刚创建 LoRA 层时：

```text
LoRA_B(LoRA_A(x)) = 0
```

因此：

```text
base + LoRA == base
```

这个性质很重要。它保证了在加载预训练 base 权重后，启用 LoRA 不会立刻引入随机扰动。训练开始后，B 矩阵逐渐从 0 被更新，LoRA 分支才开始改变模型输出。

## 10. 参数冻结

两套实现都用名称中的 `lora_` 判断参数是否属于 LoRA。

Keras 遍历 `model.weights`：

```python
for w in model.weights:
    if "lora_" not in w.path:
        w.trainable = False
    else:
        w.trainable = True
```

也就是说：

```text
路径中包含 lora_ 的权重可训练；
其他 base 参数全部冻结。
```

PyTorch 遍历 `model.named_parameters()` 并设置：

```python
param.requires_grad = "lora_" in name
```

Keras 需要在 `model.compile()` 前完成冻结；PyTorch optimizer 只接收 `requires_grad=True` 的参数。最终目标相同：基础参数冻结，只有 LoRA A/B 参数更新。

## 11. 训练入口

两个版本的 `lora_sft.py` 都执行以下主流程：

```text
1. 初始化 tokenizer
2. 构造 use_lora=True 的 pretrain model
3. 加载 base 权重
4. 冻结非 LoRA 参数
5. 构造 optimizer、SFT loss 和 accuracy
6. 读取 SFT 数据
7. 按 batch 训练并在测试集评估
8. 每个 epoch 结束保存 LoRA 权重
```

### 11.1 Keras / TensorFlow

Keras 使用无限生成器和 `model.fit()`：

```python
model = create_pretrain_model(configs)
apply_train_weights(model, weight_map_path)
mark_only_lora_as_trainable(model)
model.compile(optimizer="adam", loss=sft_loss, metrics=[SFTAccuracy()])
model.fit(..., callbacks=[Lora_Evaluate(tokenizer_tool)])
```

base 权重路径为：

```text
models/{epoch}_k2v_weights.pkl
```

`Lora_Evaluate` 是 Keras callback，在每个 epoch 结束时执行样例生成并保存独立 LoRA 权重。

### 11.2 PyTorch

PyTorch 使用 `SFTDataset` 保存未 padding 的单条样本，使用 `sft_collate_fn(pad_id)` 在 batch 内动态 padding，再通过 `DataLoader` 迭代。训练入口显式执行 forward、loss、`backward()` 和 optimizer step。

测试阶段使用 `model.eval()` 和 `torch.no_grad()`。`SFTEvaluate` 由外层 epoch 循环显式调用，负责样例生成和 LoRA 保存。base 权重路径为：

```text
models/{epoch}_k2v_weights.pt
```

## 12. LoRA 权重保存

`save_lora_weights()` 只保存名称中包含 `lora_` 的权重。

Keras 遍历 `model.weights`：

```python
if "lora_" not in path:
    continue
```

保存为 pickle 字典：

```python
{
    weight.path: weight.numpy()
}
```

例如：

```text
transformerblock_0/attention/lora_q_A/kernel
transformerblock_0/attention/lora_q_B/kernel
transformerblock_0/swiGLU/lora_gate_v_A/kernel
...
```

这样 LoRA 权重可以和 base 权重分开管理：

```text
base weights     : models/*_k2v_weights.pkl
LoRA-SFT weights : lora_sft_weights/*_lora_weights.pkl
```

PyTorch 从 `model.state_dict()` 筛选 LoRA tensor，并使用 `torch.save()` 保存：

```text
base weights     : models/*_k2v_weights.pt
LoRA-SFT weights : lora_sft_weights/*_lora_weights.pt
```

两种文件格式不能跨框架直接加载，但保存的逻辑参数集合相同。

## 13. 推理加载

两套框架的推理阶段仍然使用：

```text
Prefill_Model
Decode_Model
```

当 `use_lora=True` 且 `lora_weights_path` 存在时，模型会先加载 base 权重，再加载同框架的 LoRA 权重。Keras 的调用关系是：

```python
apply_train_weights(self.model, self.weight_map_path)
if self.use_lora and os.path.isfile(self.lora_weights_path):
    apply_lora_weights(self.model, self.lora_weights_path)
```

因此推理时实际权重组合是：

```text
base model + LoRA delta
```

Prefill 和 Decode 都会执行同样的加载逻辑，保证 prompt 阶段和单 token decode 阶段使用同一套 base + LoRA 参数。

Keras 逐个匹配权重路径并检查形状；PyTorch 执行相同的“base 后加载 LoRA”顺序，但使用 `load_state_dict(..., strict=False)` 加载 LoRA state dict。

## 14. LoRA 权重合并

如果后续要在 SFT 模型基础上继续做 DPO，可以先把 SFT LoRA 合并进 base 权重，得到一个新的 merged base：

```text
base kernel + LoRA delta -> merged kernel
```

两个版本的 `lora_sft.py` 都会在训练结束后直接调用 `merge_lora_weights()` 和 `save_model_weights()`，自动生成本框架的 merged SFT 权重。Keras 版本还保留 `merge_lora_checkpoint.py`，用于手动重新合并已有 base/LoRA checkpoint。合并流程是：

```text
1. 创建 use_lora=True 的模型
2. 加载 base 权重
3. 加载 SFT LoRA 权重
4. 调用 merge_lora_weights
5. 保存合并后的 base 权重映射
```

Keras Dense kernel 的形状是：

```text
(input_dim, output_dim)
```

而 LoRA 前向是：

```python
out = base_dense(x) + scale * lora_B(lora_A(x))
```

因此合并到 base kernel 上的增量是：

```python
delta_kernel = scale * A.kernel @ B.kernel
```

PyTorch `nn.Linear` weight 的形状是 `(output_dim, input_dim)`，对应增量为：

```python
delta_weight = scale * B.weight @ A.weight
```

两种乘法顺序来自权重存储方向差异，表示的前向变换相同。PyTorch 合并后还会把模块的 `use_lora` 设为 `False`，防止前向时再次叠加已经合入 base 的 LoRA 分支；Keras 保存 merged checkpoint 时跳过 LoRA 权重，之后应由 `use_lora=False` 的模型加载。

合并后，`save_model_weights()` 会跳过路径中包含 `lora_` 的权重，只保存已经写入 base kernel 的 merged 权重。

需要注意：merge 是原地写入 base kernel 的操作，不应该对同一个已经 merge 过的模型重复调用，否则 LoRA delta 会被重复加进去。

## 15. 当前实现的验收点

本项目中一个有效的 LoRA-SFT 权重文件应该满足：

```text
1. 权重路径全部包含 lora_
2. 数量与模型层数、LoRA 接入位置一致
3. B 矩阵不再全 0
4. 推理模型加载 LoRA 后 missing 和 shape_mismatch 数量符合预期
```

以当前小模型配置为例：

```text
num_block = 4
model dimension = 64
hidden_channels = 128
lora_rank = 4
```

每层 LoRA 权重数量：

```text
Attention: q/k/v/out 各 A、B，共 8 个
SwiGLU   : gate_v/gate_w/out 各 A、B，共 6 个
每层合计: 14 个
4 层合计: 56 个
```

对应 LoRA checkpoint 应包含 56 个 tensor：Keras 文件扩展名为 `.pkl`，PyTorch 为 `.pt`。数量正确只能说明保存范围符合预期；还应确认至少部分 B 矩阵不再全为零，才能证明 LoRA 参数实际发生了训练更新。

## 16. 小结

本项目的 LoRA-SFT 可以概括为：

```text
messages 数据
  -> prompt + answer + eos
  -> answer-only loss mask
  -> 加载 base 权重
  -> 冻结非 LoRA 参数
  -> 只训练 Attention / SwiGLU 中的 LoRA 权重
  -> 每个 epoch 结束后单独保存 LoRA 权重
  -> 训练结束后自动 merge 成新的 base 权重
  -> 推理时加载 merged 权重，或按需使用 base + LoRA
```

它不是为了训练出强效果模型，而是为了把 SFT 的关键工程链路跑通：数据构造、loss mask、参数冻结、LoRA 权重管理，以及 prefill/decode 推理阶段的 LoRA 加载。
