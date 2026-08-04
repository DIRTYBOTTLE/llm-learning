# 4.2 Tokenization（分词）

> 把文字切成小块，这是模型处理文字的第一步。

## 从你已经知道的开始

在第3章，我们假设每个词都有一个索引（比如"我"=0，"爱"=1）。

**但实际上，怎么把文字变成索引呢？这就是 Tokenization 的工作！**

---

## 一、什么是 Tokenization？

### 简单定义

```
Tokenization = 把文字切成小块（token）的过程
```

### 用例子理解

```
输入："我爱学习"

可能的切分方式：
1. 字符级：["我", "爱", "学", "习"]
2. 词级：["我", "爱", "学习"]
3. 子词级：["我", "爱", "学", "习"]（BPE算法）
```

**每个 token 都会被分配一个索引（数字）！**

---

## 二、三种分词方法

### 1. 字符级分词

**把每个字符当作一个 token**

```
"我爱学习" → ["我", "爱", "学", "习"]

词表大小：几千个（中文字符数量）
```

**优点**：
- 词表小
- 不会有未知词（OOV）

**缺点**：
- 序列太长（一个字一个token）
- 语义信息弱（单个字意思不完整）

### 2. 词级分词

**把每个词当作一个 token**

```
"我爱学习" → ["我", "爱", "学习"]

词表大小：几十万个（所有词的数量）
```

**优点**：
- 语义完整
- 序列短

**缺点**：
- 词表太大
- 会有未知词（词表外的词）

### 3. 子词级分词（BPE）

**介于字符和词之间，把词切成有意义的子词**

```
"unhappiness" → ["un", "happy", "ness"]

词表大小：几万个（常用子词的数量）
```

**优点**：
- 词表适中
- 能处理未知词
- 语义信息好

**缺点**：
- 算法复杂

**GPT 用的就是这种方法！**

---

## 三、BPE 算法详解

### 什么是 BPE？

**BPE = Byte Pair Encoding（字节对编码）**

核心思想：
1. 从字符开始
2. 找出最频繁的相邻 pair
3. 把它们合并成新 token
4. 重复直到达到目标词表大小

### 用具体例子演示

假设我们有以下文本（已经分好字符）：

```
"l o w" 出现 5 次
"l o w e r" 出现 2 次
"n e w" 出现 6 次
"n e w e r" 出现 3 次
```

**第1步：统计所有相邻 pair 的频率**

```
"l o" → 5 + 2 = 7 次
"o w" → 5 + 2 = 7 次
"w e" → 2 + 6 + 3 = 11 次
"e r" → 2 + 3 = 5 次
"n e" → 6 + 3 = 9 次
```

**第2步：合并最频繁的 pair**

最频繁的是 "w e"（11次），合并成 "we"

```
"l o w" → "l o w"
"l o w e r" → "l o w er"
"n e w" → "n we"
"n e w e r" → "n we er"
```

**第3步：重复**

继续统计，继续合并...

**第4步：直到达到目标词表大小**

### BPE 的代码实现

```python
import re
from collections import Counter

def learn_bpe(corpus, num_merges=100):
    """
    学习 BPE 合并规则
    
    参数：
    - corpus: 语料库（文本）
    - num_merges: 合并次数
    """
    
    # 第1步：把文本分成字符
    words = corpus.split()
    word_freq = Counter(words)
    
    # 把每个词分成字符
    vocab = {}
    for word, freq in word_freq.items():
        chars = list(word)
        vocab[tuple(chars)] = freq
    
    # 第2步：学习合并规则
    merges = []
    
    for i in range(num_merges):
        # 统计所有相邻 pair 的频率
        pairs = Counter()
        for chars, freq in vocab.items():
            for j in range(len(chars) - 1):
                pairs[(chars[j], chars[j+1])] += freq
        
        # 找出最频繁的 pair
        if not pairs:
            break
        
        best_pair = max(pairs, key=pairs.get)
        merges.append(best_pair)
        
        # 合并这个 pair
        new_vocab = {}
        for chars, freq in vocab.items():
            new_chars = []
            j = 0
            while j < len(chars):
                if j < len(chars) - 1 and (chars[j], chars[j+1]) == best_pair:
                    new_chars.append(chars[j] + chars[j+1])
                    j += 2
                else:
                    new_chars.append(chars[j])
                    j += 1
            new_vocab[tuple(new_chars)] = freq
        
        vocab = new_vocab
    
    return merges
```

---

## 四、实际的 Tokenizer

### 使用 Hugging Face 的 Tokenizer

```python
from transformers import GPT2Tokenizer

# 加载 GPT-2 的 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# 编码
text = "Hello, how are you?"
tokens = tokenizer.encode(text)
print(tokens)  # [15496, 11, 703, 389, 345, 30]

# 解码
decoded = tokenizer.decode(tokens)
print(decoded)  # "Hello, how are you?"

# 查看 token 详情
tokens_str = tokenizer.tokenize(text)
print(tokens_str)  # ['Hello', ',', ' how', ' are', ' you', '?']
```

### 中文的 Tokenization

```python
from transformers import BertTokenizer

# 加载 BERT 的中文 tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# 编码
text = "我爱学习"
tokens = tokenizer.encode(text)
print(tokens)  # [101, 2769, 4263, 2110, 739, 102]

# 查看 token 详情
tokens_str = tokenizer.tokenize(text)
print(tokens_str)  # ['我', '爱', '学', '习']
```

---

## 五、词表大小的影响

### 词表太大

```
优点：每个词都有对应的 token
缺点：模型参数多，训练慢
```

### 词表太小

```
优点：模型参数少，训练快
缺点：一个词可能被切成很多小块，语义信息弱
```

### 常见的词表大小

| 模型 | 词表大小 |
|------|----------|
| GPT-2 | 50,257 |
| BERT (英文) | 30,522 |
| BERT (中文) | 21,128 |
| GPT-3 | 50,257 |

---

## 六、总结

### Tokenization 的作用

```
文字 → Tokenizer → token 索引 → 模型处理
```

### 三种分词方法

| 方法 | 例子 | 优点 | 缺点 |
|------|------|------|------|
| 字符级 | "我" "爱" "学" "习" | 词表小 | 序列长 |
| 词级 | "我" "爱" "学习" | 语义完整 | 词表大 |
| 子词级 | "我" "爱" "学" "习" | 平衡 | 算法复杂 |

### GPT 用的是什么？

**GPT 用的是 BPE（子词级分词）！**

---

## 练习题

### 练习1：理解分词

请用自己的话解释：
1. 什么是 Tokenization？
2. 三种分词方法有什么区别？
3. GPT 用的是哪种方法？

### 练习2：动手试试

用 Hugging Face 的 tokenizer 对以下句子进行分词：
- "Hello, world!"
- "我爱学习"

---

## 练习答案

### 练习1答案

1. **Tokenization**：把文字切成小块（token）的过程。
2. **三种方法的区别**：
   - 字符级：每个字符是一个 token
   - 词级：每个词是一个 token
   - 子词级：介于字符和词之间，把词切成有意义的子词
3. **GPT 用的是 BPE**（子词级分词）。

### 练习2答案

```python
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

text1 = "Hello, world!"
tokens1 = tokenizer.tokenize(text1)
print(tokens1)  # ['Hello', ',', ' world', '!']

text2 = "我爱学习"
tokens2 = tokenizer.tokenize(text2)
print(tokens2)  # 可能是 ['å', 'æ', 'Ī', 'ç', 'Ī', '±', 'å', '­', '¦']（中文字符被分成字节）
```
