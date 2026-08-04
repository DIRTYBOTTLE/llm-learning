"""
3.2 循环神经网络（RNN）
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class SimpleRNN(nn.Module):
    """简单的RNN实现"""
    
    def __init__(self, input_size, hidden_size, output_size):
        """
        初始化RNN
        
        Args:
            input_size: 输入维度
            hidden_size: 隐藏层维度（记忆的大小）
            output_size: 输出维度
        """
        super().__init__()
        self.hidden_size = hidden_size
        
        # 三个权重矩阵
        self.W_xh = nn.Linear(input_size, hidden_size)   # 输入 → 隐藏
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)  # 隐藏 → 隐藏
        self.W_hy = nn.Linear(hidden_size, output_size)   # 隐藏 → 输出
        
    def forward(self, x, h_prev=None):
        """
        前向传播
        
        Args:
            x: 输入序列，形状 (batch_size, seq_len, input_size)
            h_prev: 初始记忆，形状 (batch_size, hidden_size)
        """
        batch_size, seq_len, _ = x.size()
        
        # 如果没有初始记忆，就用全0
        if h_prev is None:
            h_prev = torch.zeros(batch_size, self.hidden_size)
        
        outputs = []
        h = h_prev
        hidden_states = []
        
        # 逐个时间步处理
        for t in range(seq_len):
            x_t = x[:, t, :]  # 取出第 t 个词
            h = torch.tanh(self.W_xh(x_t) + self.W_hh(h))  # 更新记忆
            y = self.W_hy(h)  # 生成输出
            outputs.append(y)
            hidden_states.append(h.clone())
        
        # 把所有时间步的输出堆叠起来
        outputs = torch.stack(outputs, dim=1)
        hidden_states = torch.stack(hidden_states, dim=1)
        
        return outputs, h, hidden_states


def demonstrate_rnn():
    """演示RNN的工作原理"""
    
    print("=" * 50)
    print("RNN的工作原理")
    print("=" * 50)
    print()
    
    # 创建RNN
    input_size = 10
    hidden_size = 20
    output_size = 5
    
    rnn = SimpleRNN(input_size, hidden_size, output_size)
    
    # 创建输入
    batch_size = 1
    seq_len = 4
    x = torch.randn(batch_size, seq_len, input_size)
    
    print(f"输入形状: {x.shape}")
    print(f"  - batch_size: {batch_size}")
    print(f"  - seq_len: {seq_len}")
    print(f"  - input_size: {input_size}")
    print()
    
    # 前向传播
    outputs, h_final, hidden_states = rnn(x)
    
    print(f"输出形状: {outputs.shape}")
    print(f"最终记忆形状: {h_final.shape}")
    print(f"所有记忆形状: {hidden_states.shape}")
    print()
    
    # 展示RNN的计算过程
    print("RNN的计算过程：")
    print("  时间步1: h₁ = tanh(W_xh · x₁ + W_hh · h₀)")
    print("  时间步2: h₂ = tanh(W_xh · x₂ + W_hh · h₁)")
    print("  时间步3: h₃ = tanh(W_xh · x₃ + W_hh · h₂)")
    print("  时间步4: h₄ = tanh(W_xh · x₄ + W_hh · h₃)")
    print()
    
    print("关键点：")
    print("  ✓ 每个时间步使用相同的权重")
    print("  ✓ 记忆在时间步之间传递")
    print("  ✓ 每个时间步都可以产生输出")
    
    return rnn, x, outputs, hidden_states


def visualize_rnn():
    """可视化RNN的处理过程"""
    
    # 创建RNN
    rnn = SimpleRNN(input_size=5, hidden_size=10, output_size=3)
    
    # 创建输入
    x = torch.randn(1, 5, 5)
    
    # 前向传播
    outputs, h_final, hidden_states = rnn(x)
    
    # 可视化记忆的变化
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 记忆热力图
    ax1 = axes[0]
    hidden_np = hidden_states[0].detach().numpy()
    im1 = ax1.imshow(hidden_np.T, cmap='RdBu', aspect='auto')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Hidden State Dimension')
    ax1.set_title('RNN Memory Over Time')
    ax1.set_xticks(range(5))
    ax1.set_xticklabels(['t=1', 't=2', 't=3', 't=4', 't=5'])
    plt.colorbar(im1, ax=ax1)
    
    # 输出热力图
    ax2 = axes[1]
    output_np = outputs[0].detach().numpy()
    im2 = ax2.imshow(output_np.T, cmap='Blues', aspect='auto')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Output Dimension')
    ax2.set_title('RNN Output Over Time')
    ax2.set_xticks(range(5))
    ax2.set_xticklabels(['t=1', 't=2', 't=3', 't=4', 't=5'])
    plt.colorbar(im2, ax=ax2)
    
    plt.suptitle('RNN Visualization', fontsize=14)
    plt.tight_layout()
    plt.savefig('03_Transformer架构/images/rnn_basics.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("RNN可视化已保存到: 03_Transformer架构/images/rnn_basics.png")


def demonstrate_applications():
    """演示RNN的应用场景"""
    
    print()
    print("=" * 50)
    print("RNN的应用场景")
    print("=" * 50)
    print()
    
    applications = [
        ("文本生成", "今天天气 → 真 → 好 → ...", "逐步生成下一个词"),
        ("情感分析", "这部电影太棒了！→ 正面", "处理整个句子后判断"),
        ("机器翻译", "I love you → 我爱你", "编码器→向量→解码器"),
    ]
    
    for name, example, description in applications:
        print(f"{name}:")
        print(f"  示例: {example}")
        print(f"  说明: {description}")
        print()


if __name__ == "__main__":
    # 演示RNN
    rnn, x, outputs, hidden_states = demonstrate_rnn()
    
    # 可视化
    visualize_rnn()
    
    # 应用场景
    demonstrate_applications()
