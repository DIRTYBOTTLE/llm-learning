# 4.5 GPT 的生成过程

> GPT 是怎么生成文本的？一个词一个词地生成！

## 从你已经知道的开始

在上一节，你学了 GPT 是怎么训练的。

**这一节，我们来看看 GPT 是怎么生成文本的。**

---

## 一、自回归生成

### 核心思想

**自回归**：每次生成一个词，然后把它加入输入，继续生成下一个词。

```
第1步：输入 "" → 预测 "我"
第2步：输入 "我" → 预测 "爱"
第3步：输入 "我 爱" → 预测 "学习"
第4步：输入 "我 爱 学习" → 预测 "<END>"

生成结果："我爱学习"
```

### 用代码理解

```python
def generate(model, start_tokens, max_len=50):
    """自回归生成文本"""
    
    tokens = start_tokens.copy()
    
    for _ in range(max_len):
        # 输入当前序列
        input_ids = torch.tensor([tokens])
        
        # 前向传播，得到每个位置的 logits
        with torch.no_grad():
            logits = model(input_ids)  # (1, seq_len, vocab_size)
        
        # 取最后一个位置的 logits
        next_token_logits = logits[0, -1, :]  # (vocab_size,)
        
        # 选择下一个词
        next_token = select_next_token(next_token_logits)
        
        # 如果是结束符，就停止
        if next_token == END_TOKEN:
            break
        
        # 把新词加入序列
        tokens.append(next_token)
    
    return tokens
```

---

## 二、选择下一个词的方法

### 方法1：贪心解码（Greedy Decoding）

**每次都选择概率最大的词**

```python
def greedy_decode(logits):
    """贪心解码"""
    return torch.argmax(logits).item()
```

**优点**：简单，确定性

**缺点**：生成的文本可能重复、无趣

### 方法2：采样解码（Sampling）

**根据概率分布随机采样**

```python
def sample_decode(logits, temperature=1.0):
    """采样解码"""
    
    # 温度缩放
    logits = logits / temperature
    
    # 转换成概率
    probs = torch.softmax(logits, dim=-1)
    
    # 随机采样
    next_token = torch.multinomial(probs, num_samples=1).item()
    
    return next_token
```

**优点**：生成的文本更多样、更有趣

**缺点**：可能生成不合理的词

### 方法3：Top-K 采样

**只从概率最大的 K 个词中采样**

```python
def top_k_decode(logits, k=50, temperature=1.0):
    """Top-K 采样"""
    
    # 温度缩放
    logits = logits / temperature
    
    # 取最大的 K 个
    top_k_logits, top_k_indices = torch.topk(logits, k)
    
    # 转换成概率
    top_k_probs = torch.softmax(top_k_logits, dim=-1)
    
    # 采样
    idx = torch.multinomial(top_k_probs, num_samples=1).item()
    
    # 返回原始索引
    return top_k_indices[idx].item()
```

### 方法4：Top-P 采样（Nucleus Sampling）

**从累积概率达到 P 的词中采样**

```python
def top_p_decode(logits, p=0.9, temperature=1.0):
    """Top-P 采样"""
    
    # 温度缩放
    logits = logits / temperature
    
    # 转换成概率
    probs = torch.softmax(logits, dim=-1)
    
    # 排序
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    
    # 累积概率
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # 找到累积概率超过 P 的位置
    mask = cumulative_probs - sorted_probs > p
    sorted_probs[mask] = 0
    
    # 重新归一化
    sorted_probs = sorted_probs / sorted_probs.sum()
    
    # 采样
    idx = torch.multinomial(sorted_probs, num_samples=1).item()
    
    return sorted_indices[idx].item()
```

---

## 三、温度参数

### 什么是温度？

**温度（Temperature）控制生成的随机性**

```
温度高（如 1.5）：更随机，更多样
温度低（如 0.5）：更确定，更保守
温度 = 1.0：正常
```

### 温度的影响

```python
logits = torch.tensor([1.0, 2.0, 3.0, 4.0])

# 温度 = 1.0（正常）
probs = torch.softmax(logits / 1.0, dim=-1)
print(probs)  # [0.032, 0.087, 0.237, 0.644]

# 温度 = 0.5（低温度，更确定）
probs = torch.softmax(logits / 0.5, dim=-1)
print(probs)  # [0.002, 0.018, 0.117, 0.863]

# 温度 = 2.0（高温度，更随机）
probs = torch.softmax(logits / 2.0, dim=-1)
print(probs)  # [0.105, 0.160, 0.241, 0.494]
```

**温度越低，概率分布越"尖锐"，越确定！**

---

## 四、完整的生成代码

```python
def generate_text(model, tokenizer, prompt, max_len=50, 
                  temperature=1.0, top_k=50, top_p=0.9):
    """
    生成文本
    
    参数：
    - model: GPT 模型
    - tokenizer: 分词器
    - prompt: 提示文本
    - max_len: 最大生成长度
    - temperature: 温度参数
    - top_k: Top-K 参数
    - top_p: Top-P 参数
    """
    
    model.eval()
    
    # 编码提示文本
    tokens = tokenizer.encode(prompt)
    
    for _ in range(max_len):
        # 输入当前序列
        input_ids = torch.tensor([tokens])
        
        # 前向传播
        with torch.no_grad():
            logits = model(input_ids)
        
        # 取最后一个位置的 logits
        next_token_logits = logits[0, -1, :]
        
        # 选择下一个词（使用 Top-P 采样）
        next_token = top_p_decode(next_token_logits, p=top_p, temperature=temperature)
        
        # 如果是结束符，就停止
        if next_token == tokenizer.eos_token_id:
            break
        
        # 把新词加入序列
        tokens.append(next_token)
    
    # 解码
    generated_text = tokenizer.decode(tokens)
    
    return generated_text
```

---

## 五、生成的例子

### 不同温度的对比

```
提示文本："今天天气"

温度 = 0.5（保守）：
"今天天气很好，适合出去玩。"

温度 = 1.0（正常）：
"今天天气真好，阳光明媚。"

温度 = 1.5（随机）：
"今天天气不错，突然想起小时候..."
```

### 不同 Top-K 的对比

```
提示文本："今天天气"

Top-K = 10：
"今天天气很好。"

Top-K = 50：
"今天天气真好，适合出去玩。"

Top-K = 100：
"今天天气不错，突然想去海边。"
```

---

## 六、总结

### 生成的关键点

| 概念 | 说明 |
|------|------|
| 自回归生成 | 每次生成一个词，加入输入继续生成 |
| 贪心解码 | 每次选概率最大的词 |
| 采样解码 | 根据概率随机采样 |
| Top-K | 只从最大的 K 个词中采样 |
| Top-P | 从累积概率达到 P 的词中采样 |
| 温度 | 控制随机性 |

### 选择建议

```
需要确定性输出（如代码生成）：低温度，小 Top-K
需要多样性输出（如创意写作）：高温度，大 Top-K
一般情况：温度=1.0，Top-P=0.9
```

---

## 练习题

### 练习1：理解生成过程

请用自己的话解释：
1. 什么是自回归生成？
2. 贪心解码和采样解码有什么区别？
3. 温度参数的作用是什么？

### 练习2：选择方法

对于以下任务，应该选择什么生成方法？
1. 生成代码
2. 写一首诗
3. 回答事实性问题

---

## 练习答案

### 练习1答案

1. **自回归生成**：每次生成一个词，然后把它加入输入，继续生成下一个词。
2. **区别**：
   - 贪心解码：每次都选概率最大的词，确定性
   - 采样解码：根据概率随机采样，多样性
3. **温度的作用**：控制随机性，温度越高越随机，越低越确定。

### 练习2答案

1. **生成代码**：低温度，小 Top-K（需要确定性）
2. **写诗**：高温度，大 Top-K（需要创意）
3. **回答事实**：中等温度，Top-P=0.9（需要准确性）
