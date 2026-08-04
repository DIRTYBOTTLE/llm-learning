# 4.4 GPT 的训练过程

> GPT 是怎么训练的？用大量的文本，学习预测下一个词。

## 从你已经知道的开始

在第3章，你学了：
- GPT 用的是 Transformer 解码器
- GPT 的任务是预测下一个词

**这一节，我们来详细看看 GPT 是怎么训练的。**

---

## 一、训练数据

### 数据来源

GPT 的训练数据是**大量的文本**：

```
- 书籍
- 网页
- 维基百科
- 新闻
- 代码
- ...
```

GPT-3 的训练数据有 **45TB**！

### 数据格式

```
原始文本："今天天气真好，适合出去玩。"

处理后：
输入："今天天气真好"
目标："，适合出去玩。"
```

### 数据处理流程

```
1. 收集文本数据
2. 清洗数据（去除噪声）
3. Tokenization（分词）
4. 切分成训练样本
```

---

## 二、训练样本的构造

### 因果语言模型（CLM）

**训练目标**：给定前面的词，预测下一个词

```
句子："我 爱 学习 深度 学习"

构造训练样本：
样本1：输入 ""，目标 "我"
样本2：输入 "我"，目标 "爱"
样本3：输入 "我 爱"，目标 "学习"
样本4：输入 "我 爱 学习"，目标 "深度"
样本5：输入 "我 爱 学习 深度"，目标 "学习"
```

### 实际处理方式

**不是构造多个样本，而是一次处理整个序列！**

```
输入序列："我 爱 学习 深度 学习"
目标序列："爱 学习 深度 学习 <END>"

模型同时预测每个位置的下一个词！
```

### 用代码理解

```python
# 训练数据
text = "我爱学习深度学习"

# Tokenization
tokens = tokenize(text)  # [1, 2, 3, 4, 5, 3]

# 构造输入和目标
input_ids = tokens[:-1]   # [1, 2, 3, 4, 5]  （去掉最后一个）
target_ids = tokens[1:]    # [2, 3, 4, 5, 3]  （去掉第一个）

# 输入和目标错开一位！
```

---

## 三、模型结构

### GPT 的结构

```
输入 token 索引
    ↓
词嵌入 + 位置编码
    ↓
Transformer 解码器层 × N
    ↓
线性层（输出词表大小）
    ↓
softmax（得到概率分布）
```

### 用代码理解

```python
class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff):
        super().__init__()
        
        # 词嵌入 + 位置编码
        self.embedding = TokenEmbedding(vocab_size, d_model)
        
        # Transformer 解码器层
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        
        # 输出层
        self.output_layer = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        """
        x: 输入 token 索引，形状 (batch_size, seq_len)
        """
        # 词嵌入 + 位置编码
        x = self.embedding(x)  # (batch_size, seq_len, d_model)
        
        # 通过解码器层
        for layer in self.layers:
            x = layer(x)
        
        # 输出层
        logits = self.output_layer(x)  # (batch_size, seq_len, vocab_size)
        
        return logits
```

---

## 四、损失函数

### 交叉熵损失

**任务**：预测下一个词，这是一个分类问题

```
模型输出：logits = [0.1, 0.5, 0.3, 0.1]  （每个词的概率）
目标：index = 1  （正确答案是第1个词）

损失 = -log(softmax(logits)[1])
     = -log(0.5)
     = 0.693
```

### 用代码理解

```python
import torch.nn.functional as F

def compute_loss(logits, targets):
    """
    计算交叉熵损失
    
    参数：
    - logits: 模型输出，形状 (batch_size, seq_len, vocab_size)
    - targets: 目标索引，形状 (batch_size, seq_len)
    """
    # 展平
    logits = logits.view(-1, logits.size(-1))  # (batch_size * seq_len, vocab_size)
    targets = targets.view(-1)                  # (batch_size * seq_len,)
    
    # 计算交叉熵损失
    loss = F.cross_entropy(logits, targets)
    
    return loss
```

---

## 五、训练循环

### 完整的训练代码

```python
def train_gpt(model, dataloader, optimizer, num_epochs=10):
    """训练 GPT"""
    
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for batch in dataloader:
            # 获取输入和目标
            input_ids = batch["input_ids"]    # (batch_size, seq_len)
            target_ids = batch["target_ids"]  # (batch_size, seq_len)
            
            # 前向传播
            logits = model(input_ids)  # (batch_size, seq_len, vocab_size)
            
            # 计算损失
            loss = compute_loss(logits, target_ids)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 打印训练信息
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
```

---

## 六、学习率调度

### 为什么需要学习率调度？

```
训练初期：学习率大，快速学习
训练后期：学习率小，精细调整
```

### 常见的调度策略

**Warmup + Cosine Decay**

```
学习率 = lr_max × min(step/warmup_steps, 0.5 × (1 + cos(π × step / total_steps)))
```

```python
def get_lr(step, warmup_steps, total_steps, lr_max):
    """获取当前学习率"""
    if step < warmup_steps:
        # Warmup 阶段：线性增加
        return lr_max * step / warmup_steps
    else:
        # Cosine Decay 阶段：余弦衰减
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return lr_max * 0.5 * (1 + math.cos(math.pi * progress))
```

---

## 七、总结

### GPT 训练的关键点

| 环节 | 说明 |
|------|------|
| 训练数据 | 大量文本 |
| 训练目标 | 预测下一个词 |
| 损失函数 | 交叉熵 |
| 优化器 | Adam |
| 学习率调度 | Warmup + Cosine Decay |

### 训练流程

```
1. 准备训练数据（大量文本）
2. Tokenization（分词）
3. 构造训练样本（输入和目标错开一位）
4. 前向传播（得到 logits）
5. 计算损失（交叉熵）
6. 反向传播（计算梯度）
7. 更新参数（优化器）
8. 重复 4-7 直到收敛
```

---

## 练习题

### 练习1：理解训练数据

请用自己的话解释：
1. GPT 的训练数据是什么？
2. 输入和目标是怎么构造的？
3. 为什么要错开一位？

### 练习2：理解损失函数

请用自己的话解释：
1. 为什么用交叉熵损失？
2. 交叉熵损失是怎么计算的？

---

## 练习答案

### 练习1答案

1. **训练数据**：大量的文本（书籍、网页、维基百科等）。
2. **输入和目标**：输入是前面的词，目标是后面的词，错开一位。
3. **为什么错开**：因为任务是预测下一个词，所以目标是输入向后移动一位。

### 练习2答案

1. **为什么用交叉熵**：因为这是一个分类问题（从词表中选择一个词），交叉熵是分类问题的标准损失函数。
2. **怎么计算**：
   - 把 logits 通过 softmax 转换成概率
   - 取目标词对应的概率
   - 计算 -log(概率)
