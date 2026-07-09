# 端到端 Demo

`demo.py` 是本项目的本地端到端入口，用来把核心流程串起来：

```text
BBPE tokenizer
  -> pretrain
  -> LoRA-SFT
  -> LoRA-DPO
  -> merged weight inference
```

这个脚本不是为了替代各个训练入口，而是为了让读者快速看清楚本项目的完整工程链路。

## 1. 本地文件准备

仓库不包含训练语料、SFT/DPO 数据、tokenizer 配置或模型权重。运行 `demo.py` 前，需要在本地准备这些目录：

```text
data/
SFT_data/
DPO_data/
```

其中：

```text
data/      : 预训练纯文本语料
SFT_data/  : LoRA-SFT jsonl 数据
DPO_data/  : LoRA-DPO chosen/rejected jsonl 数据
```

请只使用自己有权使用的数据。学习用途不等于自动获得分发授权。

## 2. 运行方式

安装依赖后，在项目根目录运行：

```bash
python demo.py
```

脚本会按顺序检查本地文件是否存在；如果对应文件不存在，就执行相应阶段。

## 3. 主要阶段

### 3.1 BBPE

如果不存在：

```text
tokenizer_config/vocab.json
tokenizer_config/merge_rules.json
```

脚本会调用 `BBPETrainer` 从 `data/*.txt` 训练 tokenizer。

### 3.2 Pretrain

如果不存在：

```text
models/0_k2v_weights.pkl
```

脚本会创建 decoder-only Transformer，并使用 next-token prediction 做一次预训练。

### 3.3 LoRA-SFT

如果不存在：

```text
lora_sft_weights/0_k2v_lora_merged.pkl
```

脚本会加载 pretrain base 权重，开启 LoRA，只训练 LoRA 参数，然后把 SFT LoRA 合并进 base 权重。

### 3.4 LoRA-DPO

如果不存在：

```text
lora_dpo_weights/0_k2v_lora_merged.pkl
```

脚本会加载 SFT merged base，开启新的 LoRA 参数，用 DPO loss 训练 chosen/rejected 偏好数据，然后再保存 DPO merged base。

### 3.5 Inference

最后脚本会加载：

```text
lora_dpo_weights/0_k2v_lora_merged.pkl
```

并使用 `Interface` 跑 prefill/decode KVCache 推理。

## 4. 生成文件

`demo.py` 运行后可能生成：

```text
tokenizer_config/*.json
models/*
lora_sft_weights/*
lora_dpo_weights/*
```

这些文件默认被 `.gitignore` 排除，不随仓库提交。

## 5. 注意事项

- `demo.py` 默认使用很小的模型配置，主要用于验证流程，不代表模型效果。
- 如果本地已经存在某阶段的输出文件，脚本会跳过该阶段。
- LoRA-SFT 和 LoRA-DPO 都会先训练 LoRA 参数，再按需 merge 成新的 base 权重。
- 推理阶段加载的是 DPO merged base，因此不需要再额外加载 LoRA 权重。
