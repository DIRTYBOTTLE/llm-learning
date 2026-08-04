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

## 五、KV Cache（键值缓存）详解

> 这是理解 GPT 生成效率的关键技术！

### 为什么需要 KV Cache？

#### 问题：原始生成方式太慢了！

```
生成第1个词：输入 [我] → 模型计算 → 输出 "爱"
生成第2个词：输入 [我, 爱] → 模型计算 → 输出 "学习"
生成第3个词：输入 [我, 爱, 学习] → 模型计算 → 输出 "深度"
生成第4个词：输入 [我, 爱, 学习, 深度] → 模型计算 → 输出 "学习"

问题：
  生成第2个词时，"我"的注意力被重新计算了一遍！
  生成第3个词时，"我"和"爱"的注意力都被重新计算了一遍！
  生成第4个词时，"我"、"爱"、"学习"的注意力都被重新计算了一遍！

每次都重复计算之前的内容，太浪费了！
```

#### 用数字感受一下

```
假设生成100个token：

没有KV Cache：
  第1次：计算1个token的注意力
  第2次：计算2个token的注意力
  第3次：计算3个token的注意力
  ...
  第100次：计算100个token的注意力
  
  总计算量 = 1+2+3+...+100 = 5050 次

有KV Cache：
  第1次：计算1个token的注意力
  第2次：计算1个token的注意力（用缓存）
  第3次：计算1个token的注意力（用缓存）
  ...
  第100次：计算1个token的注意力（用缓存）
  
  总计算量 = 100 次

快了 50 倍！
```

---

### 回顾：注意力是怎么计算的？

在第3章学过，注意力计算需要 Q、K、V：

```
Q = 查询（我在找什么？）
K = 键（我有什么？）
V = 值（我的内容是什么？）

注意力 = softmax(Q × K^T / √d_k) × V
```

#### 关键发现

```
当输入序列是 [我, 爱, 学习] 时：

位置0("我")的K和V：
  K₀ = 我的词嵌入 × W_K
  V₀ = 我的词嵌入 × W_V

位置1("爱")的K和V：
  K₁ = 爱的词嵌入 × W_K
  V₁ = 爱的词嵌入 × W_V

位置2("学习")的K和V：
  K₂ = 学习的词嵌入 × W_K
  V₂ = 学习的词嵌入 × W_V
```

**重要发现**：当输入变成 [我, 爱, 学习, 深度] 时：

```
位置0的K₀和V₀不会变！
位置1的K₁和V₁不会变！
位置2的K₂和V₂不会变！

只有位置3的K₃和V₃是新计算的！
```

---

### KV Cache 的原理

**核心思想**：之前位置的 K 和 V 不会变，可以缓存起来重复使用！

#### 第一次生成

```
输入：[我]

步骤1：计算位置0的K、V
  K₀ = 我 × W_K = [0.2, 0.5]
  V₀ = 我 × W_V = [0.3, 0.1]

步骤2：缓存 K、V
  K_cache = [K₀] = [[0.2, 0.5]]
  V_cache = [V₀] = [[0.3, 0.1]]

步骤3：计算注意力
  Q₀ = 我 × W_Q = [0.4, 0.3]
  注意力分数 = Q₀ × K_cache^T = [0.4,0.3] × [0.2,0.5]^T = 0.23
  输出 = softmax([0.23]) × V_cache = 1.0 × [0.3, 0.1] = [0.3, 0.1]

步骤4：预测下一个词 → "爱"
```

#### 第二次生成

```
输入：[我, 爱]
但我们只计算新位置！

步骤1：只计算位置1（"爱"）的K、V
  K₁ = 爱 × W_K = [0.6, 0.2]
  V₁ = 爱 × W_V = [0.4, 0.8]

步骤2：更新缓存
  K_cache = [K₀, K₁] = [[0.2, 0.5], [0.6, 0.2]]
  V_cache = [V₀, V₁] = [[0.3, 0.1], [0.4, 0.8]]

步骤3：计算注意力（只需要计算新位置的Q）
  Q₁ = 爱 × W_Q = [0.5, 0.4]
  注意力分数 = Q₁ × K_cache^T
             = [0.5,0.4] × [[0.2,0.5],[0.6,0.2]]^T
             = [0.5,0.4] × [[0.2,0.6],[0.5,0.2]]
             = [0.5×0.2+0.4×0.5, 0.5×0.6+0.4×0.2]
             = [0.1+0.2, 0.3+0.08]
             = [0.3, 0.38]
  
  输出 = softmax([0.3, 0.38]) × V_cache
       = [0.48, 0.52] × [[0.3,0.1], [0.4,0.8]]
       = [0.48×0.3+0.52×0.4, 0.48×0.1+0.52×0.8]
       = [0.144+0.208, 0.048+0.416]
       = [0.352, 0.464]

步骤4：预测下一个词 → "学习"
```

#### 第三次生成

```
输入：[我, 爱, 学习]
但我们只计算新位置！

步骤1：只计算位置2（"学习"）的K、V
  K₂ = 学习 × W_K = [0.1, 0.7]
  V₂ = 学习 × W_V = [0.5, 0.2]

步骤2：更新缓存
  K_cache = [K₀, K₁, K₂] = [[0.2,0.5], [0.6,0.2], [0.1,0.7]]
  V_cache = [V₀, V₁, V₂] = [[0.3,0.1], [0.4,0.8], [0.5,0.2]]

步骤3：计算注意力（只需要计算新位置的Q）
  Q₂ = 学习 × W_Q = [0.3, 0.6]
  注意力分数 = Q₂ × K_cache^T = ...
  
  输出 = ...

步骤4：预测下一个词 → "<END>"
```

---

### 用图理解 KV Cache

```
┌─────────────────────────────────────────────────────────────┐
│                    KV Cache 工作流程                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  第1次生成：                                                │
│    输入: [我]                                               │
│    计算: K₀, V₀（新计算）                                   │
│    缓存: K_cache = [K₀], V_cache = [V₀]                    │
│    输出: "爱"                                               │
│                                                             │
│  第2次生成：                                                │
│    输入: [我, 爱]                                           │
│    计算: K₁, V₁（只计算新的！）                             │
│    缓存: K_cache = [K₀, K₁], V_cache = [V₀, V₁]           │
│    输出: "学习"                                             │
│                                                             │
│  第3次生成：                                                │
│    输入: [我, 爱, 学习]                                     │
│    计算: K₂, V₂（只计算新的！）                             │
│    缓存: K_cache = [K₀, K₁, K₂], V_cache = [V₀, V₁, V₂]   │
│    输出: "<END>"                                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### KV Cache 的优缺点

#### 优点

```
1. 速度快
   每次只需要计算1个新token的K、V
   而不是重新计算所有token
   
2. 计算量小
   生成100个token：100次计算（有缓存）
   而不是5050次计算（无缓存）
```

#### 缺点

```
1. 占用显存
   需要存储所有位置的K、V
   序列越长，占用显存越多
   
   例如：
     d_model = 768
     序列长度 = 2048
     每个K或V = 768 × 4字节 = 3KB
     总占用 = 2048 × 3KB × 2(K和V) × 12(层数) = 144MB
     
2. 有最大长度限制
   显存有限，不能无限存储
   超过最大长度需要截断或压缩
```

---

### 长文本怎么办？

#### 方法1：截断（丢掉旧的）

```
最大长度 = 512

生成第1-512个token：正常生成
生成第513个token：
  丢掉第1个token的K、V
  缓存变成：[K₁, K₂, ..., K₅₁₂, K₅₁₃]
  
这样缓存长度始终是512
```

#### 方法2：滑动窗口

```
只保留最近的窗口：

窗口大小 = 512

生成第1000个token时：
  只保留最近512个token的K、V
  丢掉前面的

优点：显存占用固定
缺点：丢失了早期的上下文
```

#### 方法3：使用长上下文模型

```
一些模型专门优化了长上下文：

GPT-4：支持 128K token（约10万字）
Claude：支持 200K token（约15万字）
Gemini：支持 1M token（约75万字）

这些模型使用了特殊技术：
  - 更高效的注意力机制
  - 压缩历史信息
  - 分层缓存
```

---

### KV Cache 的代码实现

```python
class GPTWithKVCache(nn.Module):
    """带KV Cache的GPT模型"""
    
    def __init__(self, vocab_size, d_model, num_heads, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads)
            for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        # KV Cache
        self.k_cache = [None] * num_layers  # 每层一个缓存
        self.v_cache = [None] * num_layers
    
    def reset_cache(self):
        """清空缓存（开始新的生成）"""
        self.k_cache = [None] * len(self.layers)
        self.v_cache = [None] * len(self.layers)
    
    def forward(self, x, use_cache=True):
        """
        x: 输入token，形状 (batch_size, seq_len)
           如果use_cache=True，seq_len通常=1（只输入新token）
        """
        x = self.embedding(x)
        
        for i, layer in enumerate(self.layers):
            if use_cache and self.k_cache[i] is not None:
                # 有缓存：只计算新token的K、V
                # 然后和缓存拼接
                k_new, v_new = layer.compute_kv(x)
                k = torch.cat([self.k_cache[i], k_new], dim=1)
                v = torch.cat([self.v_cache[i], v_new], dim=1)
                
                # 更新缓存
                self.k_cache[i] = k
                self.v_cache[i] = v
                
                # 计算注意力（用完整的K、V，但只计算新token的Q）
                x = layer.attention(x, k, v)
            else:
                # 没有缓存：正常计算
                x = layer(x)
                if use_cache:
                    # 保存K、V到缓存
                    self.k_cache[i], self.v_cache[i] = layer.get_kv()
        
        return self.output_layer(x)


def generate_with_cache(model, start_token, max_len=50):
    """使用KV Cache生成文本"""
    
    model.reset_cache()  # 清空缓存
    
    tokens = [start_token]
    
    for _ in range(max_len):
        # 输入只有最后一个token（因为有缓存）
        input_ids = torch.tensor([[tokens[-1]]])
        
        # 前向传播（使用缓存）
        logits = model(input_ids, use_cache=True)
        
        # 预测下一个词
        next_token = torch.argmax(logits[0, -1]).item()
        
        tokens.append(next_token)
    
    return tokens
```

---

### 一句话总结

| 问题 | 答案 |
|------|------|
| KV Cache是什么？ | 缓存之前位置的K、V，避免重复计算 |
| 为什么需要？ | 让生成速度更快 |
| 缺点是什么？ | 占用显存，有最大长度限制 |
| 长文本怎么办？ | 截断、滑动窗口、或用长上下文模型 |

---

## 六、生成的例子

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

## 七、总结

### 生成的关键点

| 概念 | 说明 |
|------|------|
| 自回归生成 | 每次生成一个词，加入输入继续生成 |
| 贪心解码 | 每次选概率最大的词 |
| 采样解码 | 根据概率随机采样 |
| Top-K | 只从最大的 K 个词中采样 |
| Top-P | 从累积概率达到 P 的词中采样 |
| 温度 | 控制随机性 |
| KV Cache | 缓存K、V，加速生成 |

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

### 练习2：理解 KV Cache

请用自己的话解释：
1. 为什么需要 KV Cache？
2. KV Cache 的原理是什么？
3. KV Cache 有什么缺点？

### 练习3：选择方法

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

1. **为什么需要**：避免重复计算之前位置的K、V，加快生成速度
2. **原理**：缓存之前位置的K、V，每次只计算新token的K、V，然后和缓存拼接
3. **缺点**：占用显存，序列越长占用越多

### 练习3答案

1. **生成代码**：低温度，小 Top-K（需要确定性）
2. **写诗**：高温度，大 Top-K（需要创意）
3. **回答事实**：中等温度，Top-P=0.9（需要准确性）
