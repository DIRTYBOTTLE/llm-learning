# 第5章 预训练与微调

> 训练一个大模型需要什么？这章从头到尾讲清楚。

## 学习目标

学完这章，你能：
- 理解预训练的完整流程
- 掌握学习率调度的数学原理
- 理解 LoRA 的低秩分解原理
- 掌握 RLHF 和 DPO 的核心思想

## 章节内容

### 5.1 预训练数据

数据准备：
- 数据清洗、去重
- 数据配比
- 数据质量评估

### 5.2 预训练流程

训练循环：
```python
for batch in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

梯度累积：
```
loss = loss / accumulation_steps
loss.backward()
if (step + 1) % accumulation_steps == 0:
    optimizer.step()
    optimizer.zero_grad()
```

### 5.3 学习率调度

Warmup + Cosine Decay：
```
lr = lr_max * min(step/warmup_steps, 0.5 * (1 + cos(π * step / total_steps)))
```

### 5.4 有监督微调 SFT

指令微调数据格式：
```
{
  "instruction": "请翻译成英文",
  "input": "今天天气真好",
  "output": "The weather is nice today"
}
```

### 5.5 参数高效微调

LoRA 的低秩分解：
```
W = W₀ + BA
其中 B ∈ ℝ^(d×r), A ∈ ℝ^(r×k), r << min(d, k)
```

参数量对比：
- 全量微调：d × k 参数
- LoRA：r × (d + k) 参数

### 5.6 RLHF 与 DPO

奖励模型：
```
R(x, y) = 模型对 (x, y) 的评分
```

PPO 算法：
```
L_PPO = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]
```

DPO 损失：
```
L_DPO = -log σ(β log π(y_w|x)/π_ref(y_w|x) - β log π(y_l|x)/π_ref(y_l|x))
```

## 前置知识

- 第4章：语言模型、预训练目标
- 第2章：优化器、学习率调度

## 预计学习时间

- 理论学习：6-8 小时
- 代码实践：4-6 小时
- 练习巩固：3 小时

---

*待完善*
