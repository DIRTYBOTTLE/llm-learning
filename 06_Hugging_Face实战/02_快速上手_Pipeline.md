# 6.2 快速上手：Pipeline

> Pipeline 让你一行代码完成各种 AI 任务！

## 从你已经知道的开始

在上一节，你学了：
- Hugging Face 提供了大量预训练模型
- Transformers 是最流行的 NLP 库

**这一节，我们来学习最简单的使用方式：Pipeline！**

---

## 一、什么是 Pipeline？

### 简介

```
Pipeline 是 Transformers 库中最简单的使用方式

特点：
  - 一行代码完成任务
  - 自动处理分词、模型调用、后处理
  - 不需要了解模型细节
  
类比：
  Pipeline 就像一个"黑盒子"
  你只需要输入文本，它就输出结果
```

### Pipeline 的工作流程

```
输入文本
    ↓
分词（Tokenization）
    ↓
模型推理（Model Inference）
    ↓
后处理（Post-processing）
    ↓
输出结果
```

---

## 二、文本生成

### 使用 Pipeline 生成文本

```python
from transformers import pipeline

# 创建文本生成器
generator = pipeline("text-generation", model="gpt2")

# 生成文本
result = generator("Today the weather is")
print(result)

# 输出：
# [{'generated_text': 'Today the weather is nice and I want to go outside...'}]
```

### 设置生成参数

```python
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

# 设置参数
result = generator(
    "Today the weather is",
    max_length=50,        # 最大长度
    num_return_sequences=3,  # 生成3个结果
    temperature=0.7,      # 温度
    do_sample=True        # 是否采样
)

for i, r in enumerate(result):
    print(f"结果{i+1}: {r['generated_text']}")
```

### 使用中文模型

```python
from transformers import pipeline

# 使用中文 GPT 模型
generator = pipeline("text-generation", model="uer/gpt2-chinese-cluecorpussmall")

result = generator("今天天气")
print(result)
```

---

## 三、情感分析

### 使用 Pipeline 进行情感分析

```python
from transformers import pipeline

# 创建情感分析器
classifier = pipeline("sentiment-analysis")

# 分析情感
result = classifier("This movie is great!")
print(result)

# 输出：
# [{'label': 'POSITIVE', 'score': 0.9998}]
```

### 批量分析

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")

# 批量分析
texts = [
    "I love this movie!",
    "This is terrible.",
    "It's okay, nothing special."
]

results = classifier(texts)

for text, result in zip(texts, results):
    print(f"文本: {text}")
    print(f"情感: {result['label']}, 置信度: {result['score']:.4f}")
    print()
```

### 使用中文情感分析模型

```python
from transformers import pipeline

# 使用中文情感分析模型
classifier = pipeline("sentiment-analysis", model="uer/roberta-base-finetuned-chinanews-chinese")

result = classifier("这部电影很好看！")
print(result)
```

---

## 四、问答系统

### 使用 Pipeline 进行问答

```python
from transformers import pipeline

# 创建问答器
qa_pipeline = pipeline("question-answering")

# 问答
result = qa_pipeline(
    question="What is the capital of China?",
    context="China is a country in East Asia. Its capital is Beijing."
)

print(result)

# 输出：
# {'score': 0.99, 'start': 45, 'end': 51, 'answer': 'Beijing'}
```

### 使用示例

```python
from transformers import pipeline

qa_pipeline = pipeline("question-answering")

context = """
The Amazon rainforest, also known as Amazonia, is a moist broadleaf tropical 
rainforest in the Amazon biome that covers most of the Amazon basin of South America. 
This basin encompasses 7,000,000 km2 (2,700,000 sq mi), of which 5,500,000 km2 
(2,100,000 sq mi) are covered by the rainforest.
"""

questions = [
    "What is the Amazon rainforest?",
    "How large is the Amazon basin?",
    "Where is the Amazon rainforest located?"
]

for question in questions:
    result = qa_pipeline(question=question, context=context)
    print(f"问题: {question}")
    print(f"答案: {result['answer']}")
    print(f"置信度: {result['score']:.4f}")
    print()
```

---

## 五、文本分类

### 使用 Pipeline 进行文本分类

```python
from transformers import pipeline

# 创建分类器
classifier = pipeline("zero-shot-classification")

# 分类
result = classifier(
    "This is a course about Python programming",
    candidate_labels=["education", "politics", "business"]
)

print(result)

# 输出：
# {'labels': ['education', 'politics', 'business'], 
#  'scores': [0.95, 0.03, 0.02]}
```

### 使用示例

```python
from transformers import pipeline

classifier = pipeline("zero-shot-classification")

text = "Apple announced the new iPhone yesterday"

candidate_labels = ["technology", "sports", "politics", "entertainment"]

result = classifier(text, candidate_labels=candidate_labels)

print(f"文本: {text}")
print("分类结果:")
for label, score in zip(result['labels'], result['scores']):
    print(f"  {label}: {score:.4f}")
```

---

## 六、翻译

### 使用 Pipeline 进行翻译

```python
from transformers import pipeline

# 创建翻译器（英语→法语）
translator = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")

# 翻译
result = translator("Hello, how are you?")
print(result)

# 输出：
# [{'translation_text': 'Bonjour, comment allez-vous?'}]
```

### 中英翻译

```python
from transformers import pipeline

# 英译中
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-zh")

result = translator("Hello, how are you?")
print(result)

# 中译中
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-zh-en")

result = translator("你好，你怎么样？")
print(result)
```

---

## 七、其他任务

### 文本摘要

```python
from transformers import pipeline

summarizer = pipeline("summarization")

text = """
The Amazon rainforest, also known as Amazonia, is a moist broadleaf tropical 
rainforest in the Amazon biome that covers most of the Amazon basin of South America. 
This basin encompasses 7,000,000 km2 (2,700,000 sq mi), of which 5,500,000 km2 
(2,100,000 sq mi) are covered by the rainforest. The rainforest was created during 
the Eocene era. It appeared following a global reduction of tropical temperatures 
when the Atlantic Ocean had expanded sufficiently to provide a warm, moist climate 
to the Amazon basin.
"""

result = summarizer(text, max_length=50, min_length=20)
print(result[0]['summary_text'])
```

### 命名实体识别（NER）

```python
from transformers import pipeline

ner = pipeline("ner", grouped_entities=True)

text = "My name is John and I live in New York."

result = ner(text)
print(result)

# 输出：
# [{'entity_group': 'PER', 'score': 0.99, 'word': 'John'},
#  {'entity_group': 'LOC', 'score': 0.99, 'word': 'New York'}]
```

---

## 八、Pipeline 支持的任务

```
┌─────────────────────────────────────────────────────────────┐
│                    Pipeline 支持的任务                       │
├─────────────────────────────────────────────────────────────┤
│  任务                  说明                                 │
├─────────────────────────────────────────────────────────────┤
│  text-generation       文本生成                             │
│  sentiment-analysis    情感分析                             │
│  question-answering    问答                                 │
│  zero-shot-classification 零样本分类                        │
│  translation           翻译                                 │
│  summarization         摘要                                 │
│  ner                   命名实体识别                         │
│  fill-mask             完形填空                             │
│  feature-extraction    特征提取                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 九、总结

### Pipeline 的关键点

| 概念 | 说明 |
|------|------|
| Pipeline | 最简单的使用方式 |
| 优点 | 一行代码完成任务 |
| 缺点 | 不够灵活，不适合自定义 |
| 适用场景 | 快速原型、测试 |

### 使用建议

```
1. 快速原型：用 Pipeline
   简单、快速、不需要了解细节
   
2. 生产环境：用 AutoClass
   更灵活、更可控
   
3. 微调训练：用 AutoClass
   需要更多控制
```

---

## 练习题

### 练习1：文本生成

使用 Pipeline 生成一段关于"人工智能"的文本。

### 练习2：情感分析

使用 Pipeline 分析以下句子的情感：
- "I love this product!"
- "This is the worst experience ever."
- "It's okay, nothing special."

---

## 练习答案

### 练习1答案

```python
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")
result = generator("Artificial intelligence is", max_length=50)
print(result[0]['generated_text'])
```

### 练习2答案

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")

texts = [
    "I love this product!",
    "This is the worst experience ever.",
    "It's okay, nothing special."
]

for text in texts:
    result = classifier(text)
    print(f"{text}: {result[0]['label']} ({result[0]['score']:.4f})")
```
