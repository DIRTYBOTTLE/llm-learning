"""
3.3 RNN 的三个大问题
"""

import torch
import numpy as np
import matplotlib.pyplot as plt


def demonstrate_gradient_vanishing():
    """演示梯度消失问题"""
    
    print("=" * 50)
    print("问题一：梯度消失")
    print("=" * 50)
    print()
    
    seq_len = 20
    decay_factor = 0.9
    
    # 计算梯度衰减
    gradients = [1.0]
    for t in range(1, seq_len):
        gradients.append(gradients[-1] * decay_factor)
    
    print("假设每个时间步梯度衰减 0.9：")
    print(f"  时间步1:  {gradients[0]:.6f}")
    print(f"  时间步10: {gradients[9]:.6f}")
    print(f"  时间步20: {gradients[19]:.6f}")
    print()
    print("结论：梯度从 1.0 衰减到了 0.12，几乎消失了！")
    
    # 可视化
    plt.figure(figsize=(8, 4))
    plt.plot(range(seq_len), gradients, 'b-', linewidth=2, marker='o')
    plt.xlabel('Time Step')
    plt.ylabel('Gradient')
    plt.title('Gradient Vanishing Problem')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('03_Transformer架构/images/gradient_vanishing.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n梯度消失可视化已保存")


def demonstrate_sequential_computation():
    """演示顺序计算瓶颈"""
    
    print()
    print("=" * 50)
    print("问题二：不能并行计算")
    print("=" * 50)
    print()
    
    print("RNN 必须按顺序计算：")
    print("  时间步1: 计算 h₁（需要 h₀ 和 x₁）")
    print("  时间步2: 计算 h₂（需要 h₁）← 必须等 h₁ 算完")
    print("  时间步3: 计算 h₃（需要 h₂）← 必须等 h₂ 算完")
    print("  ...")
    print()
    print("CNN 可以并行计算：")
    print("  所有卷积核同时计算")
    print()
    print("影响：RNN 训练速度慢")


def demonstrate_long_distance_dependency():
    """演示长距离依赖问题"""
    
    print()
    print("=" * 50)
    print("问题三：记不住太远的东西")
    print("=" * 50)
    print()
    
    sentence = ["我", "昨天", "去了", "北京", "那里", "很冷", "所以", "我", "买了", "羽绒服"]
    
    print(f"句子：{' '.join(sentence)}")
    print()
    print("需要捕捉的关系：")
    print("  '北京' ↔ '那里'（相隔3个词）")
    print("  '很冷' ↔ '羽绒服'（相隔4个词）")
    print()
    
    # 模拟信息衰减
    decay_rate = 0.8
    info_remaining = [1.0]
    for t in range(1, len(sentence)):
        info_remaining.append(info_remaining[-1] * decay_rate)
    
    print("信息衰减：")
    for i, (word, info) in enumerate(zip(sentence, info_remaining)):
        print(f"  位置{i}: {word} → 信息剩余 {info:.2%}")
    print()
    print("结论：位置0的'我'的信息，到位置9只剩下 13%！")
    
    # 可视化
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(sentence)), info_remaining, color='steelblue', alpha=0.7)
    plt.xlabel('Position')
    plt.ylabel('Information Remaining')
    plt.title('Long-distance Dependency Problem')
    plt.xticks(range(len(sentence)), sentence, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('03_Transformer架构/images/long_distance_dependency.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("长距离依赖问题可视化已保存")


def summarize_problems():
    """总结三个问题"""
    
    print()
    print("=" * 50)
    print("总结：RNN 的三个大问题")
    print("=" * 50)
    print()
    
    print("| 问题 | 原因 | 影响 |")
    print("|------|------|------|")
    print("| 梯度消失 | 梯度经过太多时间步 | 无法学习长距离依赖 |")
    print("| 不能并行 | 必须按顺序计算 | 训练速度慢 |")
    print("| 记不住远的东西 | 信息经过太多步会衰减 | 句子越长效果越差 |")
    print()
    print("解决方案：注意力机制 + Transformer（下一节学习）")


if __name__ == "__main__":
    demonstrate_gradient_vanishing()
    demonstrate_sequential_computation()
    demonstrate_long_distance_dependency()
    summarize_problems()
