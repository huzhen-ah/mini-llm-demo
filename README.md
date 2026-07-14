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
| [PyTorch](pytorch-mini-llm/) | 已完成预训练、验证与采样、checkpoint、Prefill / Decode 和 KVCache 推理；LoRA-SFT、LoRA-DPO 与端到端 demo 待完成 |

## 目录结构

```text
.
├── keras-mini-llm/       # Keras / TensorFlow 完整版本
├── pytorch-mini-llm/     # PyTorch 对照实现
└── README.md
```

两个版本各自保持完整、可独立运行的工程结构。少量重复代码用于保留清晰的一一对应关系，便于逐文件比较两个框架的实现差异。

详细的运行方式、文件说明和后续计划见各版本目录中的 README。
