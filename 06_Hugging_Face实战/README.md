# 第6章 Hugging Face 实战

> Hugging Face 是 AI 的"应用商店"。这章教你开箱即用。

## 学习目标

学完这章，你能：
- 熟练使用 Transformers 库
- 掌握 Datasets 库的数据处理
- 使用 Trainer API 训练模型
- 完成一个完整的微调项目

## 章节内容

### 6.1 Transformers 库

Pipeline API：
```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("I love this movie!")
```

### 6.2 Datasets 库

数据加载和预处理：
```python
from datasets import load_dataset

dataset = load_dataset("imdb")
```

### 6.3 Trainer API

训练参数配置：
```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
)
trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
trainer.train()
```

### 6.4 完整微调项目

实战：情感分析
- 数据准备
- 模型选择
- 训练配置
- 评估和部署

## 前置知识

- 第5章：预训练与微调
- Python 基础

## 预计学习时间

- 理论学习：2-3 小时
- 代码实践：3-4 小时
- 练习巩固：2 小时

---

*待完善*
