# Mini LLM Demo

这是一个面向学习与教学的 mini LLM 项目，计划使用 Keras / TensorFlow 和 PyTorch 分别实现同一条完整链路，方便对照两个框架中的模型结构、训练流程与推理实现。

## 两个版本

| 版本 | 状态 | 说明 |
| --- | --- | --- |
| [keras-mini-llm](keras-mini-llm/) | 已完成 | Keras / TensorFlow 实现，覆盖 BBPE、预训练、LoRA-SFT、LoRA-DPO 和 KVCache 推理 |
| [pytorch-mini-llm](pytorch-mini-llm/) | 进行中 | PyTorch 对应实现 |

## 目录结构

```text
.
├── README.md
├── keras-mini-llm/       # Keras / TensorFlow 完整版本
└── pytorch-mini-llm/     # PyTorch 版本
```

两个版本各自保持完整、可独立阅读的工程结构。这样会保留少量重复代码，但更适合逐文件对照学习，也避免读者为了理解一个版本而跨目录寻找公共实现。

当前 Keras 版本的运行方式、项目结构和技术说明见：[keras-mini-llm/README.md](keras-mini-llm/README.md)。
