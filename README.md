# Mini LLM Demo

这是一个面向学习与教学的 mini LLM 项目，分别使用 Keras / TensorFlow 和 PyTorch 实现同一条训练与推理链路，用于对照两个框架中的模型结构、训练流程、权重管理和 KVCache 实现。

## 项目链路

```text
BBPE tokenizer
  -> decoder-only Transformer pretraining
  -> LoRA-SFT
  -> LoRA-DPO
  -> Prefill / decode
  -> KVCache inference
  -> end-to-end demo
```

## 实现进度

| 版本 | 当前状态 |
| --- | --- |
| [Keras / TensorFlow](keras-mini-llm/) | 已完成预训练、LoRA-SFT、LoRA-DPO、Prefill / Decode、KVCache 推理和端到端 demo |
| [PyTorch](pytorch-mini-llm/) | 已完成预训练、LoRA-SFT、LoRA-DPO、Prefill / Decode、KVCache 推理和端到端 demo |

当前 Keras / TensorFlow 与 PyTorch 两个版本的核心训练和推理链路均已跑通。两套版本的运行说明已分别整理在各自 README 中，共用原理与实现差异统一记录在根目录 `docs/`。

## 目录结构

```text
.
├── keras-mini-llm/       # Keras / TensorFlow 完整版本
├── pytorch-mini-llm/     # PyTorch 对照实现
├── docs/                 # 两套框架共用的原理与实现文档
└── README.md
```

两个版本各自保持完整、可独立运行的工程结构。少量重复代码用于保留清晰的一一对应关系，便于逐文件比较两个框架的实现差异。

详细的运行方式和文件说明见各版本目录中的 README；通用实现文档见下方索引。

## 通用实现文档

- [Mini LLM 模型架构](docs/model_architecture.md)
- [Multi-Head Self-Attention 原理与实现](docs/attention.md)
- [BBPE 原理与实现](docs/bbpe.md)
- [RoPE 原理与实现](docs/rope.md)
- [KVCache 原理与实现](docs/kvcache.md)
- [LoRA-SFT 原理与实现](docs/lora_sft.md)
- [LoRA-DPO 原理与实现](docs/lora_dpo.md)
- [推理流程说明](docs/inference.md)
- [端到端 Demo](docs/demo.md)

全部根目录实现文档均已按 Keras / PyTorch 当前代码校订；共用原理合并说明，框架差异在对应章节中分别记录。
