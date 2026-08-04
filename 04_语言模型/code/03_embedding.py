"""
4.3 词嵌入演示

这个文件演示：
1. 词嵌入的基本概念
2. 词嵌入层的实现
3. 位置编码
4. 词向量的语义
"""

import torch
import torch.nn as nn
import numpy as np


def demonstrate_embedding_basic():
    """
    演示词嵌入的基本概念
    """
    print("=" * 60)
    print("词嵌入的基本概念")
    print("=" * 60)
    print()
    
    # 假设的词表
    vocab = {
        "我": 0,
        "爱": 1,
        "学习": 2,
        "深度": 3,
        "机器": 4,
        "学习_": 5
    }
    
    print("词表：")
    for word, idx in vocab.items():
        print(f"  {word}: {idx}")
    print()
    
    # 词嵌入：把索引转换成向量
    print("词嵌入的作用：")
    print("-" * 40)
    print("索引只是数字，没有语义信息")
    print("词嵌入把索引转换成有意义的向量")
    print()
    
    # 示例：随机生成词向量（实际中这些向量是学习得到的）
    d_model = 8  # 向量维度
    
    print(f"示例（向量维度 = {d_model}）：")
    for word, idx in vocab.items():
        # 随机生成向量（模拟）
        vector = np.random.randn(d_model).round(2)
        print(f"  {word:4s} -> {vector}")
    print()


def demonstrate_embedding_layer():
    """
    演示 PyTorch 的 Embedding 层
    """
    print("=" * 60)
    print("PyTorch Embedding 层演示")
    print("=" * 60)
    print()
    
    # 参数
    vocab_size = 1000  # 词表大小
    d_model = 64       # 向量维度
    
    # 创建 Embedding 层
    embedding = nn.Embedding(vocab_size, d_model)
    
    print(f"创建 Embedding 层：")
    print(f"  词表大小: {vocab_size}")
    print(f"  向量维度: {d_model}")
    print(f"  参数数量: {vocab_size * d_model}")
    print()
    
    # 输入：词的索引
    word_indices = torch.tensor([0, 1, 2, 3, 4])
    
    print(f"输入（词的索引）: {word_indices.tolist()}")
    
    # 前向传播：获取词向量
    word_vectors = embedding(word_indices)
    
    print(f"输出（词向量）形状: {word_vectors.shape}")
    print(f"  -> 5 个词，每个词 {d_model} 维向量")
    print()
    
    # 查看某个词的向量
    print("查看第一个词的向量（前5维）：")
    print(f"  {word_vectors[0, :5].tolist()}")
    print()
    
    return embedding


def demonstrate_positional_encoding():
    """
    演示位置编码
    """
    print("=" * 60)
    print("位置编码演示")
    print("=" * 60)
    print()
    
    print("为什么需要位置编码？")
    print("-" * 40)
    print("'我爱学习' 和 '学习爱我'")
    print("词相同，但顺序不同，意思完全不同！")
    print("词嵌入只能表示词的意思，不能表示位置")
    print("所以需要位置编码来补充位置信息")
    print()
    
    # 位置编码公式
    print("位置编码公式：")
    print("  PE(pos, 2i) = sin(pos / 10000^(2i/d_model))")
    print("  PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))")
    print()
    
    # 实现位置编码
    d_model = 64
    max_len = 10
    
    # 创建位置编码矩阵
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    
    # 计算频率
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                         (-torch.log(torch.tensor(10000.0)) / d_model))
    
    # 应用公式
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    print("位置编码示例（前3个位置，前4维）：")
    print("-" * 40)
    for pos in range(3):
        print(f"位置 {pos}: {pe[pos, :4].tolist()}")
    print()
    
    # 可视化位置编码的特点
    print("位置编码的特点：")
    print("  - 不同位置有不同的编码")
    print("  - 相邻位置的编码比较相似")
    print("  - 编码是确定性的（不是学习得到的）")
    print()
    
    return pe


def demonstrate_embedding_with_position():
    """
    演示词嵌入 + 位置编码
    """
    print("=" * 60)
    print("词嵌入 + 位置编码")
    print("=" * 60)
    print()
    
    # 参数
    vocab_size = 100
    d_model = 16
    seq_len = 4
    
    # 创建 Embedding 层
    word_embedding = nn.Embedding(vocab_size, d_model)
    
    # 创建位置编码
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                         (-torch.log(torch.tensor(10000.0)) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    # 输入：词的索引
    input_indices = torch.tensor([0, 1, 2, 3])  # 假设是 "我 爱 学习 深度"
    
    print(f"输入（词的索引）: {input_indices.tolist()}")
    print()
    
    # 1. 词嵌入
    word_emb = word_embedding(input_indices)
    print(f"词嵌入形状: {word_emb.shape}")
    print(f"词嵌入（第一个词，前4维）: {word_emb[0, :4].tolist()}")
    print()
    
    # 2. 位置编码
    print(f"位置编码形状: {pe.shape}")
    print(f"位置编码（位置0，前4维）: {pe[0, :4].tolist()}")
    print()
    
    # 3. 最终输入 = 词嵌入 + 位置编码
    final_input = word_emb + pe
    print(f"最终输入形状: {final_input.shape}")
    print(f"最终输入（第一个词，前4维）: {final_input[0, :4].tolist()}")
    print()
    
    print("最终输入 = 词嵌入 + 位置编码")
    print("这样，每个词既有语义信息，又有位置信息！")
    print()
    
    return final_input


def demonstrate_word_vector_semantics():
    """
    演示词向量的语义
    """
    print("=" * 60)
    print("词向量的语义")
    print("=" * 60)
    print()
    
    print("相似的词，向量也相似：")
    print("-" * 40)
    
    # 模拟词向量（实际中这些是学习得到的）
    word_vectors = {
        "国王": np.array([0.8, 0.2, 0.9, 0.1]),
        "女王": np.array([0.7, 0.3, 0.8, 0.2]),
        "男人": np.array([0.6, 0.1, 0.7, 0.0]),
        "女人": np.array([0.5, 0.2, 0.6, 0.1]),
        "苹果": np.array([0.1, 0.9, 0.2, 0.8]),
        "橘子": np.array([0.2, 0.8, 0.3, 0.7]),
    }
    
    for word, vec in word_vectors.items():
        print(f"  {word}: {vec}")
    print()
    
    # 计算相似度（余弦相似度）
    def cosine_similarity(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    print("词向量相似度：")
    print("-" * 40)
    
    pairs = [
        ("国王", "女王"),
        ("国王", "男人"),
        ("苹果", "橘子"),
        ("国王", "苹果"),
    ]
    
    for word1, word2 in pairs:
        sim = cosine_similarity(word_vectors[word1], word_vectors[word2])
        print(f"  {word1} - {word2}: {sim:.4f}")
    print()
    
    print("著名的词向量运算：")
    print("-" * 40)
    print("  国王 - 男人 + 女人 ≈ 女王")
    print()
    print("这说明词向量学到了语义关系！")
    print()


def main():
    """主函数"""
    
    print("\n" + "=" * 60)
    print("第4章 第3节：词嵌入演示")
    print("=" * 60 + "\n")
    
    # 1. 词嵌入的基本概念
    demonstrate_embedding_basic()
    
    print("-" * 60 + "\n")
    
    # 2. PyTorch Embedding 层
    demonstrate_embedding_layer()
    
    print("-" * 60 + "\n")
    
    # 3. 位置编码
    demonstrate_positional_encoding()
    
    print("-" * 60 + "\n")
    
    # 4. 词嵌入 + 位置编码
    demonstrate_embedding_with_position()
    
    print("-" * 60 + "\n")
    
    # 5. 词向量的语义
    demonstrate_word_vector_semantics()
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
