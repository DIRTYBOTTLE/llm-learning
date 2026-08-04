"""
4.4 GPT 训练过程演示

这个文件演示：
1. 训练数据的准备
2. 训练样本的构造
3. 损失函数的计算
4. 训练循环
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def demonstrate_training_data():
    """
    演示训练数据的准备
    """
    print("=" * 60)
    print("训练数据的准备")
    print("=" * 60)
    print()
    
    # 原始文本
    corpus = [
        "我爱学习深度学习",
        "深度学习很有趣",
        "我爱机器学习",
        "机器学习是人工智能的一部分"
    ]
    
    print("原始语料库：")
    for i, text in enumerate(corpus):
        print(f"  {i+1}. '{text}'")
    print()
    
    # Tokenization（简化版）
    print("Tokenization（分词）：")
    print("-" * 40)
    
    # 简单的字符级分词
    all_chars = set()
    for text in corpus:
        all_chars.update(list(text))
    
    # 创建词表
    vocab = {char: idx for idx, char in enumerate(sorted(all_chars))}
    vocab["<PAD>"] = len(vocab)
    vocab["<START>"] = len(vocab)
    vocab["<END>"] = len(vocab)
    
    print(f"词表大小: {len(vocab)}")
    print(f"词表（前10个）: {list(vocab.items())[:10]}")
    print()
    
    return corpus, vocab


def demonstrate_training_samples():
    """
    演示训练样本的构造
    """
    print("=" * 60)
    print("训练样本的构造")
    print("=" * 60)
    print()
    
    # 原始句子
    sentence = "我爱学习"
    tokens = list(sentence)
    
    print(f"原始句子: '{sentence}'")
    print(f"分词结果: {tokens}")
    print()
    
    # 构造训练样本
    print("构造训练样本（因果语言模型）：")
    print("-" * 40)
    
    # 方法1：逐个构造
    print("方法1：逐个构造")
    for i in range(len(tokens)):
        input_text = tokens[:i] if i > 0 else ["<START>"]
        target = tokens[i]
        print(f"  输入: {input_text} -> 目标: {target}")
    print()
    
    # 方法2：一次处理整个序列
    print("方法2：一次处理整个序列（错开一位）")
    input_tokens = ["<START>"] + tokens[:-1]
    target_tokens = tokens + ["<END>"]
    
    print(f"  输入序列: {input_tokens}")
    print(f"  目标序列: {target_tokens}")
    print()
    
    print("关键点：输入和目标错开一位！")
    print("  - 输入[0] = <START> -> 目标[0] = '我'")
    print("  - 输入[1] = '我' -> 目标[1] = '爱'")
    print("  - 输入[2] = '爱' -> 目标[2] = '学'")
    print("  - ...")
    print()
    
    return input_tokens, target_tokens


def demonstrate_loss_function():
    """
    演示损失函数的计算
    """
    print("=" * 60)
    print("损失函数的计算")
    print("=" * 60)
    print()
    
    print("任务：预测下一个词（分类问题）")
    print("-" * 40)
    print()
    
    # 模拟模型输出
    vocab_size = 10
    seq_len = 3
    
    # logits：模型的原始输出
    logits = torch.randn(seq_len, vocab_size)
    print(f"模型输出 logits 形状: {logits.shape}")
    print(f"  -> {seq_len} 个位置，每个位置 {vocab_size} 个词的分数")
    print()
    
    # 目标：正确的下一个词的索引
    targets = torch.tensor([2, 5, 7])  # 假设正确的词索引
    print(f"目标索引: {targets.tolist()}")
    print()
    
    # 计算交叉熵损失
    print("计算交叉熵损失：")
    print("-" * 40)
    
    # 展平 logits 和 targets
    logits_flat = logits.view(-1, vocab_size)  # (seq_len, vocab_size)
    targets_flat = targets.view(-1)            # (seq_len,)
    
    # 计算损失
    loss = F.cross_entropy(logits_flat, targets_flat)
    
    print(f"损失值: {loss.item():.4f}")
    print()
    
    # 解释损失的含义
    print("损失的含义：")
    print("  - 损失越小，模型预测越准确")
    print("  - 损失 = -log(P(正确答案))")
    print("  - 如果 P(正确答案) = 1，损失 = 0")
    print("  - 如果 P(正确答案) = 0.5，损失 = 0.693")
    print()
    
    # 演示概率和损失的关系
    print("概率和损失的关系：")
    print("-" * 40)
    for prob in [0.9, 0.5, 0.1]:
        loss_val = -np.log(prob)
        print(f"  P(正确答案) = {prob:.1f} -> 损失 = {loss_val:.4f}")
    print()
    
    return loss


def demonstrate_training_loop():
    """
    演示训练循环
    """
    print("=" * 60)
    print("训练循环演示")
    print("=" * 60)
    print()
    
    # 创建一个简单的模型
    class SimpleLM(nn.Module):
        def __init__(self, vocab_size, d_model):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.linear = nn.Linear(d_model, vocab_size)
        
        def forward(self, x):
            # x: (batch_size, seq_len)
            emb = self.embedding(x)  # (batch_size, seq_len, d_model)
            logits = self.linear(emb)  # (batch_size, seq_len, vocab_size)
            return logits
    
    # 参数
    vocab_size = 20
    d_model = 16
    batch_size = 2
    seq_len = 5
    
    # 创建模型
    model = SimpleLM(vocab_size, d_model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("模型结构：")
    print(f"  词表大小: {vocab_size}")
    print(f"  向量维度: {d_model}")
    print()
    
    # 模拟训练数据
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    target_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print("训练数据：")
    print(f"  输入形状: {input_ids.shape}")
    print(f"  目标形状: {target_ids.shape}")
    print()
    
    # 训练循环
    print("训练循环：")
    print("-" * 40)
    
    num_epochs = 3
    
    for epoch in range(num_epochs):
        # 前向传播
        logits = model(input_ids)  # (batch_size, seq_len, vocab_size)
        
        # 计算损失
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            target_ids.view(-1)
        )
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch + 1}: Loss = {loss.item():.4f}")
    
    print()
    print("训练完成！损失在下降，说明模型在学习！")
    print()
    
    return model


def demonstrate_learning_rate_schedule():
    """
    演示学习率调度
    """
    print("=" * 60)
    print("学习率调度")
    print("=" * 60)
    print()
    
    print("为什么需要学习率调度？")
    print("-" * 40)
    print("  训练初期：学习率大，快速学习")
    print("  训练后期：学习率小，精细调整")
    print()
    
    # Warmup + Cosine Decay
    def get_lr(step, warmup_steps, total_steps, lr_max):
        if step < warmup_steps:
            # Warmup 阶段：线性增加
            return lr_max * step / warmup_steps
        else:
            # Cosine Decay 阶段：余弦衰减
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return lr_max * 0.5 * (1 + np.cos(np.pi * progress))
    
    # 参数
    warmup_steps = 100
    total_steps = 1000
    lr_max = 0.001
    
    print("学习率调度示例：")
    print(f"  Warmup 步数: {warmup_steps}")
    print(f"  总步数: {total_steps}")
    print(f"  最大学习率: {lr_max}")
    print()
    
    # 显示学习率变化
    steps = [0, 50, 100, 200, 500, 800, 1000]
    
    print("学习率变化：")
    for step in steps:
        lr = get_lr(step, warmup_steps, total_steps, lr_max)
        print(f"  步骤 {step:4d}: 学习率 = {lr:.6f}")
    print()
    
    print("特点：")
    print("  - 前100步：学习率线性增加（Warmup）")
    print("  - 100步后：学习率余弦衰减")
    print("  - 这样训练更稳定！")


def main():
    """主函数"""
    
    print("\n" + "=" * 60)
    print("第4章 第4节：GPT 训练过程演示")
    print("=" * 60 + "\n")
    
    # 1. 训练数据的准备
    corpus, vocab = demonstrate_training_data()
    
    print("-" * 60 + "\n")
    
    # 2. 训练样本的构造
    input_tokens, target_tokens = demonstrate_training_samples()
    
    print("-" * 60 + "\n")
    
    # 3. 损失函数的计算
    demonstrate_loss_function()
    
    print("-" * 60 + "\n")
    
    # 4. 训练循环
    demonstrate_training_loop()
    
    print("-" * 60 + "\n")
    
    # 5. 学习率调度
    demonstrate_learning_rate_schedule()
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
