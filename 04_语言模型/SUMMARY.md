# 第4章 语言模型 - 内容总结

## 章节结构

### 4.1 什么是语言模型
- **理论文件**：`01_什么是语言模型.md`
- **代码文件**：`code/01_language_model_concept.py`
- **核心概念**：
  - 语言模型的任务：预测下一个词
  - 自回归生成
  - 概率链式法则
  - 训练目标

### 4.2 Tokenization（分词）
- **理论文件**：`02_Tokenization.md`
- **代码文件**：`code/02_tokenization.py`
- **核心概念**：
  - 字符级分词
  - 词级分词
  - 子词级分词（BPE）
  - Token 到索引的转换
  - OOV 问题

### 4.3 词嵌入
- **理论文件**：`03_词嵌入.md`
- **代码文件**：`code/03_embedding.py`
- **核心概念**：
  - Embedding 层
  - 位置编码
  - 词向量的语义
  - 向量维度选择

### 4.4 GPT 训练过程
- **理论文件**：`04_GPT训练过程.md`
- **代码文件**：`code/04_gpt_training.py`
- **核心概念**：
  - 训练数据准备
  - 训练样本构造
  - 交叉熵损失
  - 训练循环
  - 学习率调度

### 4.5 GPT 生成过程
- **理论文件**：`05_GPT生成过程.md`
- **代码文件**：`code/05_gpt_generation.py`
- **核心概念**：
  - 自回归生成
  - 贪心解码
  - 采样解码
  - Top-K 采样
  - Top-P 采样
  - 温度参数

### 4.6 实战：用 GPT 生成文本
- **理论文件**：`06_实战_GPT生成.md`
- **代码文件**：`code/01_gpt_demo.py`
- **核心概念**：
  - 使用 Hugging Face
  - 加载预训练模型
  - 生成文本
  - 调整参数

## 代码文件列表

| 文件名 | 描述 |
|--------|------|
| `01_language_model_concept.py` | 语言模型概念演示 |
| `02_tokenization.py` | 分词方法演示 |
| `03_embedding.py` | 词嵌入演示 |
| `04_gpt_training.py` | GPT 训练过程演示 |
| `05_gpt_generation.py` | GPT 生成过程演示 |
| `01_gpt_demo.py` | 使用 GPT 生成文本 |
| `generate_images.py` | 生成本章图片 |

## 图片资源

| 图片名 | 描述 |
|--------|------|
| `tokenization_comparison.png` | 分词方法对比 |
| `embedding_visualization.png` | 词嵌入可视化 |
| `temperature_effect.png` | 温度参数效果 |
| `autoregressive_generation.png` | 自回归生成过程 |

## 学习路径

```
4.1 什么是语言模型
    ↓
4.2 Tokenization（分词）
    ↓
4.3 词嵌入
    ↓
4.4 GPT 训练过程
    ↓
4.5 GPT 生成过程
    ↓
4.6 实战：用 GPT 生成文本
```

## 与第3章的联系

- 第3章介绍了 Transformer 架构（编码器和解码器）
- 第4章专注于语言模型（使用 Transformer 解码器）
- GPT 就是一个基于 Transformer 解码器的语言模型

## 与第5章的联系

- 第4章介绍了 GPT 的基本训练和生成
- 第5章将介绍预训练与微调
- 预训练 = 在大规模数据上训练语言模型
- 微调 = 在特定任务上调整模型
