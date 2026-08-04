"""
3.1 序列数据是什么
"""

import torch
import numpy as np
import matplotlib.pyplot as plt


def demonstrate_sequence_data():
    """演示序列数据的特点"""
    
    print("=" * 50)
    print("序列数据是什么")
    print("=" * 50)
    print()
    
    # 1. 文本序列
    print("1. 文本序列：")
    print("   '我爱学习' vs '学习爱我'")
    print("   词相同，但顺序不同，意思完全不同！")
    print()
    
    # 2. 时间序列
    print("2. 时间序列：")
    temperatures = [20, 22, 25, 23, 21, 19, 18]
    print(f"   过去7天的气温: {temperatures}")
    print("   数据点的顺序很重要，不能打乱！")
    print()
    
    # 3. 序列数据的特点
    print("3. 序列数据的特点：")
    print("   ✓ 顺序性：数据点的顺序很重要")
    print("   ✓ 可变长度：不同序列的长度可能不同")
    print("   ✓ 上下文依赖：当前数据与前后数据相关")
    print()
    
    return temperatures


def demonstrate_cnn_limitation():
    """演示CNN处理序列数据的局限性"""
    
    print("=" * 50)
    print("为什么CNN处理不好序列数据")
    print("=" * 50)
    print()
    
    print("问题1：CNN 不懂顺序")
    print("  对于CNN来说，'我爱学习'和'学习爱我'可能差不多")
    print("  但实际上，它们的意思完全不同！")
    print()
    
    print("问题2：CNN 看不远")
    print("  CNN的卷积核很小，只能看到局部的几个词")
    print("  但文字中，相隔很远的词可能有关系")
    print()
    
    print("问题3：CNN 不能处理变长序列")
    print("  图片大小固定（比如28×28），但文字长度是变化的")
    print()
    
    print("结论：我们需要一种新的模型来处理序列数据！")
    print("答案：循环神经网络（RNN）")
    print()


def visualize_sequence_data():
    """可视化序列数据"""
    
    # 创建示例数据
    sentences = [
        ["我", "爱", "学习"],
        ["学习", "爱", "我"],
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    for idx, (sentence, ax) in enumerate(zip(sentences, axes)):
        # 创建词向量（随机）
        word_vectors = np.random.randn(len(sentence), 5)
        
        # 绘制热力图
        im = ax.imshow(word_vectors, cmap='RdBu', aspect='auto')
        ax.set_xticks(range(5))
        ax.set_xticklabels(['dim1', 'dim2', 'dim3', 'dim4', 'dim5'])
        ax.set_yticks(range(len(sentence)))
        ax.set_yticklabels(sentence)
        ax.set_title(f'Sentence {idx+1}: {" ".join(sentence)}')
        
        # 添加数值标签
        for i in range(len(sentence)):
            for j in range(5):
                ax.text(j, i, f'{word_vectors[i][j]:.1f}', 
                       ha='center', va='center', fontsize=8)
    
    plt.suptitle('Sequence Data: Same Words, Different Order', fontsize=14)
    plt.tight_layout()
    plt.savefig('03_Transformer架构/images/sequence_data.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("序列数据可视化已保存到: 03_Transformer架构/images/sequence_data.png")


if __name__ == "__main__":
    # 演示序列数据
    temperatures = demonstrate_sequence_data()
    
    # 演示CNN的局限性
    demonstrate_cnn_limitation()
    
    # 可视化
    visualize_sequence_data()
