"""
3.4 注意力机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt


class SimpleAttention(nn.Module):
    """简单的注意力机制"""
    
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        
    def forward(self, X):
        Q = self.W_Q(X)
        K = self.W_K(X)
        V = self.W_V(X)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_model)
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights


def demonstrate_attention_intuition():
    """演示注意力机制的直觉"""
    
    print("=" * 50)
    print("注意力机制的直觉")
    print("=" * 50)
    print()
    
    print("类比：人看东西")
    print("  - 看照片时，你会关注人脸（重要），而不是背景（不重要）")
    print("  - 读文章时，你会关注关键词，而不是每个字")
    print()
    
    print("处理文字也是一样：")
    print("  句子：'我昨天去了北京，那里很冷'")
    print("  当看到'很冷'时：")
    print("    - 应该关注'北京'（知道是哪里冷）")
    print("    - 不需要关注'昨天'（时间不重要）")


def demonstrate_attention_calculation():
    """演示注意力的计算过程"""
    
    print()
    print("=" * 50)
    print("注意力的计算过程")
    print("=" * 50)
    print()
    
    print("核心公式：Attention(Q, K, V) = softmax(QKᵀ / √d_k) V")
    print()
    
    print("三个角色：")
    print("  Q (Query)：我想找什么？")
    print("  K (Key)：每个位置有什么？")
    print("  V (Value)：每个位置的内容")
    print()
    
    print("计算步骤：")
    print("  1. 计算相似度：scores = Q × Kᵀ")
    print("  2. 缩放：scores = scores / √d_k")
    print("  3. 归一化：weights = softmax(scores)")
    print("  4. 加权求和：output = weights × V")
    
    # 创建示例
    batch_size = 1
    seq_len = 4
    d_model = 8
    
    X = torch.randn(batch_size, seq_len, d_model)
    attention = SimpleAttention(d_model)
    
    output, weights = attention(X)
    
    print()
    print(f"输入形状: {X.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {weights.shape}")
    
    # 可视化注意力权重
    plt.figure(figsize=(6, 5))
    weights_np = weights[0].detach().numpy()
    plt.imshow(weights_np, cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(label='Attention Weight')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.title('Attention Weights')
    plt.tight_layout()
    plt.savefig('03_Transformer架构/images/attention_weights.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n注意力权重可视化已保存")
    
    return output, weights


def compare_rnn_and_attention():
    """对比RNN和注意力机制"""
    
    print()
    print("=" * 50)
    print("RNN vs 注意力机制")
    print("=" * 50)
    print()
    
    print("| 特性 | RNN | 注意力 |")
    print("|------|-----|--------|")
    print("| 信息传递 | 一步步传 | 直接连接 |")
    print("| 长距离依赖 | 难捕捉 | 容易捕捉 |")
    print("| 并行计算 | 不能 | 能 |")
    print("| 可解释性 | 差 | 好 |")
    print()
    
    print("为什么注意力能解决RNN的问题：")
    print("  1. 梯度消失：梯度直接传回去，不用经过中间步骤")
    print("  2. 不能并行：所有位置可以同时计算")
    print("  3. 长距离依赖：任意两个位置可以直接联系")


if __name__ == "__main__":
    demonstrate_attention_intuition()
    output, weights = demonstrate_attention_calculation()
    compare_rnn_and_attention()
