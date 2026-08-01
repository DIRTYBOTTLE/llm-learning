# 第3章 Transformer 架构

> 为什么 ChatGPT 这么厉害？因为用了 Transformer。

## 学习目标

学完这章，你能：
- 理解 Transformer 的完整架构
- 掌握自注意力机制的数学原理
- 理解多头注意力、位置编码的作用
- 用 Transformer 实现文本分类

## 章节内容

### 3.1 从 RNN 到 Transformer

为什么需要 Transformer？
- RNN 的梯度消失问题
- RNN 的顺序计算瓶颈
- Transformer 的并行计算优势

### 3.2 自注意力机制

核心公式：
```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) V

其中：
- Q = XW_Q (Query 矩阵)
- K = XW_K (Key 矩阵)
- V = XW_V (Value 矩阵)
- d_k: Key 的维度
```

为什么要除以 √d_k？
- 防止点积过大导致 softmax 梯度消失

### 3.3 多头注意力

多个"视角"同时看：
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
其中 head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

### 3.4 位置编码

像给座位"编号"：
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### 3.5 Transformer 前馈网络

每个位置独立处理：
```
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
```

### 3.6 用 Transformer 做分类

实战：情感分析
- 输入：一段影评
- 输出：正面/负面评价

## 前置知识

- 第1章：神经网络、激活函数、损失函数
- 第2章：批归一化、优化器

## 预计学习时间

- 理论学习：4-6 小时
- 代码实践：3-4 小时
- 练习巩固：2 小时

---

*待完善*
