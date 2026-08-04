# 4.6 实战：用 GPT 生成文本

> 动手实践，用 GPT 生成文本！

## 从你已经知道的开始

前几节你学了：
- 语言模型是什么
- Tokenization 怎么做
- 词嵌入怎么工作
- GPT 怎么训练和生成

**这一节，我们来动手实践！**

---

## 一、使用 Hugging Face 的 GPT

### 安装依赖

```bash
pip install transformers torch
```

### 加载模型

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 加载 GPT-2 模型和 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

print("模型加载完成！")
```

### 生成文本

```python
def generate_text(prompt, max_length=100):
    """生成文本"""
    
    # 编码输入
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    
    # 生成
    outputs = model.generate(
        inputs,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )
    
    # 解码输出
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text

# 测试
prompt = "Today the weather is"
result = generate_text(prompt)
print(result)
```

---

## 二、使用中文 GPT

### 加载中文模型

```python
from transformers import BertTokenizer, GPT2LMHeadModel

# 使用中文 tokenizer
tokenizer = BertTokenizer.from_pretrained('uer/gpt2-chinese-cluecorpussmall')
model = GPT2LMHeadModel.from_pretrained('uer/gpt2-chinese-cluecorpussmall')

print("中文模型加载完成！")
```

### 生成中文文本

```python
def generate_chinese_text(prompt, max_length=100):
    """生成中文文本"""
    
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    
    outputs = model.generate(
        inputs,
        max_length=max_length,
        num_return_sequences=1,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text

# 测试
prompt = "今天天气"
result = generate_chinese_text(prompt)
print(result)
```

---

## 三、调整参数

### 温度的影响

```python
# 低温度：更确定
result_low = generate_text("Today", temperature=0.3)

# 中温度：正常
result_mid = generate_text("Today", temperature=0.7)

# 高温度：更随机
result_high = generate_text("Today", temperature=1.2)

print("低温度:", result_low)
print("中温度:", result_mid)
print("高温度:", result_high)
```

### Top-K 的影响

```python
# 小 Top-K：更保守
result_small = generate_text("Today", top_k=10)

# 大 Top-K：更多样
result_large = generate_text("Today", top_k=100)

print("小 Top-K:", result_small)
print("大 Top-K:", result_large)
```

---

## 四、完整示例

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class TextGenerator:
    """文本生成器"""
    
    def __init__(self, model_name='gpt2'):
        """初始化"""
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()
    
    def generate(self, prompt, max_length=100, temperature=0.7, 
                 top_k=50, top_p=0.95):
        """生成文本"""
        
        # 编码输入
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                do_sample=True,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                num_return_sequences=1
            )
        
        # 解码输出
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated_text
    
    def generate_multiple(self, prompt, num_sequences=3, **kwargs):
        """生成多个结果"""
        
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                num_return_sequences=num_sequences,
                **kwargs
            )
        
        results = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            results.append(text)
        
        return results

# 使用示例
generator = TextGenerator()

# 生成单个结果
result = generator.generate("Today the weather is", max_length=50)
print("生成结果:", result)

# 生成多个结果
results = generator.generate_multiple("I love", num_sequences=3, max_length=30)
for i, result in enumerate(results):
    print(f"结果{i+1}: {result}")
```

---

## 五、总结

### 使用 GPT 生成文本的步骤

```
1. 加载模型和 tokenizer
2. 编码输入文本
3. 调用 generate 方法
4. 解码输出文本
```

### 关键参数

| 参数 | 作用 | 建议值 |
|------|------|--------|
| max_length | 最大生成长度 | 50-200 |
| temperature | 随机性 | 0.7-1.0 |
| top_k | Top-K 采样 | 50 |
| top_p | Top-P 采样 | 0.95 |

---

## 练习题

### 练习1：动手实践

使用 Hugging Face 的 GPT-2，生成以下提示的续写：
- "Once upon a time"
- "The meaning of life is"
- "Python is a programming language that"

### 练习2：调整参数

对比不同参数生成的结果：
- 温度：0.3, 0.7, 1.2
- Top-K：10, 50, 100

---

## 练习答案

### 练习1答案

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

prompts = [
    "Once upon a time",
    "The meaning of life is",
    "Python is a programming language that"
]

for prompt in prompts:
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=50, do_sample=True, top_k=50)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"提示: {prompt}")
    print(f"生成: {result}")
    print()
```

### 练习2答案

```python
prompt = "Today the weather is"

for temp in [0.3, 0.7, 1.2]:
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=30, temperature=temp, do_sample=True)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"温度 {temp}: {result}")
```
