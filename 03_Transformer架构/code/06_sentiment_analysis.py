"""
3.6 实战：电影评论情感分析
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import re
from collections import Counter


# ============ 数据准备 ============

def create_sample_data():
    """创建示例数据"""
    positive_texts = [
        "This movie is absolutely wonderful!",
        "Great acting and fantastic storyline.",
        "A masterpiece of cinema.",
        "I loved every moment of this film.",
        "Excellent performances from the cast.",
        "One of the best movies I've ever seen.",
        "The cinematography is breathtaking.",
        "A truly heartwarming story.",
        "Brilliant script and outstanding direction.",
        "I couldn't stop smiling throughout.",
    ]
    
    negative_texts = [
        "This movie was a complete waste of time.",
        "Terrible acting and a boring plot.",
        "I fell asleep halfway through.",
        "The worst film I've seen this year.",
        "Don't waste your money on this.",
        "The story makes no sense at all.",
        "Poor direction and awful script.",
        "I regret watching this movie.",
        "The characters are uninteresting.",
        "A disappointing experience.",
    ]
    
    texts = positive_texts + negative_texts
    labels = [1] * len(positive_texts) + [0] * len(negative_texts)
    
    return texts, labels


def build_vocab(texts):
    """构建词汇表"""
    word_counts = Counter()
    for text in texts:
        tokens = re.findall(r'\b\w+\b', text.lower())
        word_counts.update(tokens)
    
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for idx, (word, _) in enumerate(word_counts.items(), start=2):
        vocab[word] = idx
    
    return vocab


class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=20):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        tokens = re.findall(r'\b\w+\b', text.lower())
        indices = [self.vocab.get(t, self.vocab['<UNK>']) for t in tokens]
        
        if len(indices) < self.max_len:
            indices += [0] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]
        
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)


# ============ 模型定义 ============

class TransformerEncoder(nn.Module):
    """Transformer编码器"""
    
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_len=5000, dropout=0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # 位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        x = self.embedding(x)
        x = x + self.pe[:, :x.size(1), :]
        x = self.transformer(x)
        return x


class SentimentClassifier(nn.Module):
    """情感分类器"""
    
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, num_classes):
        super().__init__()
        self.encoder = TransformerEncoder(vocab_size, d_model, num_heads, num_layers, d_ff)
        self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        encoded = self.encoder(x)
        pooled = encoded.mean(dim=1)
        logits = self.classifier(pooled)
        return logits


# ============ 训练和测试 ============

def train_model():
    """训练模型"""
    print("=" * 50)
    print("训练 Transformer 情感分类器")
    print("=" * 50)
    print()
    
    texts, labels = create_sample_data()
    vocab = build_vocab(texts)
    
    print(f"训练数据: {len(texts)} 条")
    print(f"词汇表大小: {len(vocab)}")
    print()
    
    dataset = TextDataset(texts, labels, vocab, max_len=20)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    model = SentimentClassifier(
        vocab_size=len(vocab),
        d_model=32,
        num_heads=4,
        num_layers=2,
        d_ff=64,
        num_classes=2
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("开始训练...")
    for epoch in range(20):
        total_loss = 0
        correct = 0
        total = 0
        
        for texts_batch, labels_batch in dataloader:
            outputs = model(texts_batch)
            loss = criterion(outputs, labels_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels_batch).sum().item()
            total += labels_batch.size(0)
        
        if (epoch + 1) % 5 == 0:
            accuracy = 100 * correct / total
            print(f"  Epoch {epoch+1}: Loss={total_loss:.4f}, Accuracy={accuracy:.1f}%")
    
    print()
    return model, vocab


def test_model(model, vocab):
    """测试模型"""
    print("=" * 50)
    print("测试结果")
    print("=" * 50)
    print()
    
    test_texts = [
        "This movie is amazing!",
        "I hated this terrible film.",
        "The story was boring.",
        "A beautiful and touching story.",
    ]
    
    for text in test_texts:
        tokens = re.findall(r'\b\w+\b', text.lower())
        indices = [vocab.get(t, vocab['<UNK>']) for t in tokens]
        
        if len(indices) < 20:
            indices += [0] * (20 - len(indices))
        else:
            indices = indices[:20]
        
        input_tensor = torch.tensor([indices], dtype=torch.long)
        
        with torch.no_grad():
            output = model(input_tensor)
            prediction = torch.argmax(output, dim=1).item()
            confidence = torch.softmax(output, dim=1)[0][prediction].item()
        
        sentiment = "正面" if prediction == 1 else "负面"
        print(f"  '{text}'")
        print(f"  → {sentiment} ({confidence:.1%})")
        print()


if __name__ == "__main__":
    model, vocab = train_model()
    test_model(model, vocab)
