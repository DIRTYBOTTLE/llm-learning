# 大模型从零教程

从数学原理到代码实现，从网络训练到模型调优，手把手带你从零构建大语言模型。

## 适用人群

- 完全零基础，无需机器学习背景
- 想深入理解 LLM 原理的开发者
- 希望在 Apple M1 上实践的工程师

## 环境要求

- Python 3.9+
- Apple M1 / 16GB RAM（或同等配置）
- 依赖见 `requirements.txt`

```bash
pip install -r requirements.txt
```

## 学习路线

```
数学基础 → 神经网络 → Transformer → Tokenization → 预训练
                                                      ↓
                                              Hugging Face 实战
                                                      ↓
                            推理优化 ← 模型对齐 ← 高效微调
                                ↓
                          评估与 Benchmark
                                ↓
                           综合项目
```

## 目录结构

| 章节 | 内容 | 代码 |
|------|------|------|
| [00_数学基础](00_数学基础/) | 线性代数、微积分、概率论 | 3 个 |
| [01_神经网络基础](01_神经网络基础/) | 感知机、前馈网络、反向传播、优化器 | 4 个 |
| [02_Transformer架构](02_Transformer架构/) | 注意力机制、位置编码、GPT 架构 | 5 个 |
| [03_Tokenization](03_Tokenization/) | BPE 算法、SentencePiece | 2 个 |
| [04_预训练](04_预训练/) | 训练循环、学习率调度、混合精度 | 4 个 |
| [05_Hugging_Face实战](05_Hugging_Face实战/) | Transformers、Datasets、Trainer | 3 个 |
| [06_高效微调](06_高效微调/) | LoRA、QLoRA、Adapter | 3 个 |
| [07_模型对齐](07_模型对齐/) | SFT、RLHF、DPO | 2 个 |
| [08_推理优化](08_推理优化/) | KV Cache、量化、推测解码 | 3 个 |
| [09_评估与Benchmark](09_评估与Benchmark/) | 评测基准、自定义评测 | 2 个 |
| [10_综合项目](10_综合项目/) | 端到端完整项目 | 1 个 |

## 双轨教学

每个核心概念提供两种实现：

- **`xxx_from_scratch.py`** — 纯 PyTorch 手写，理解底层原理
- **框架实现** — Hugging Face 生态，掌握工程实践

## M1 适配

所有代码针对 Apple M1 / MPS 后端优化：

- 使用 `torch.device("mps")` 后端
- 手写示例控制在 10M-50M 参数
- QLoRA 微调可在 16GB 内存上运行 7B 模型
- 小 batch + 梯度累积模拟大 batch 训练

## 使用方式

1. 按章节顺序学习，每章包含 Markdown 文档和可运行代码
2. 每个 `.py` 文件可独立运行：`python xxx.py`
3. 建议先读文档理解原理，再运行代码验证

## 进度

- [ ] README
- [ ] 第0章 数学基础
- [ ] 第1章 神经网络基础
- [ ] 第2章 Transformer 架构
- [ ] 第3章 Tokenization
- [ ] 第4章 预训练
- [ ] 第5章 Hugging Face 实战
- [ ] 第6章 高效微调
- [ ] 第7章 模型对齐
- [ ] 第8章 推理优化
- [ ] 第9章 评估与 Benchmark
- [ ] 第10章 综合项目
