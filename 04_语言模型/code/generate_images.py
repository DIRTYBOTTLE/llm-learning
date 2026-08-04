"""
为第4章生成图片
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


def plot_tokenization_comparison():
    """
    绘制分词方法对比图
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    text = "我爱学习深度学习"
    
    # 字符级分词
    char_tokens = list(text)
    axes[0].barh(range(len(char_tokens)), [1]*len(char_tokens), color='skyblue')
    axes[0].set_yticks(range(len(char_tokens)))
    axes[0].set_yticklabels(char_tokens)
    axes[0].set_title('字符级分词')
    axes[0].set_xlabel('Token')
    
    # 词级分词
    word_tokens = ["我", "爱", "学习", "深度", "学习"]
    axes[1].barh(range(len(word_tokens)), [1]*len(word_tokens), color='lightgreen')
    axes[1].set_yticks(range(len(word_tokens)))
    axes[1].set_yticklabels(word_tokens)
    axes[1].set_title('词级分词')
    axes[1].set_xlabel('Token')
    
    # 子词级分词（BPE）
    subword_tokens = ["我", "爱", "学", "习", "深", "度", "学", "习"]
    axes[2].barh(range(len(subword_tokens)), [1]*len(subword_tokens), color='lightcoral')
    axes[2].set_yticks(range(len(subword_tokens)))
    axes[2].set_yticklabels(subword_tokens)
    axes[2].set_title('子词级分词 (BPE)')
    axes[2].set_xlabel('Token')
    
    plt.suptitle('三种分词方法对比', fontsize=14)
    plt.tight_layout()
    plt.savefig('04_语言模型/images/tokenization_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("已生成: tokenization_comparison.png")


def plot_embedding_visualization():
    """
    绘制词嵌入可视化图
    """
    # 模拟词向量
    words = ["国王", "女王", "男人", "女人", "苹果", "橘子"]
    vectors = np.array([
        [0.8, 0.2],  # 国王
        [0.7, 0.3],  # 女王
        [0.6, 0.1],  # 男人
        [0.5, 0.2],  # 女人
        [0.1, 0.9],  # 苹果
        [0.2, 0.8],  # 橘子
    ])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制散点
    colors = ['red', 'red', 'blue', 'blue', 'green', 'green']
    for i, (word, vec) in enumerate(zip(words, vectors)):
        ax.scatter(vec[0], vec[1], c=colors[i], s=200, alpha=0.6)
        ax.annotate(word, (vec[0], vec[1]), fontsize=12, 
                   ha='center', va='bottom')
    
    # 绘制箭头表示关系
    ax.annotate('', xy=vectors[1], xytext=vectors[0],
               arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.annotate('', xy=vectors[3], xytext=vectors[2],
               arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    
    ax.set_xlabel('维度 1', fontsize=12)
    ax.set_ylabel('维度 2', fontsize=12)
    ax.set_title('词向量空间可视化\n相似的词，向量也相似', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.6, label='皇室'),
        Patch(facecolor='blue', alpha=0.6, label='性别'),
        Patch(facecolor='green', alpha=0.6, label='水果')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('04_语言模型/images/embedding_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("已生成: embedding_visualization.png")


def plot_temperature_effect():
    """
    绘制温度参数效果图
    """
    vocab = ["我", "爱", "学习", "吃", "饭", "你"]
    logits = torch.tensor([0.1, 0.5, 0.3, 0.05, 0.03, 0.02])
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    temperatures = [0.5, 1.0, 2.0]
    
    for ax, temp in zip(axes, temperatures):
        scaled_logits = logits / temp
        probs = F.softmax(scaled_logits, dim=0).numpy()
        
        bars = ax.bar(vocab, probs, color=['skyblue', 'lightgreen', 'lightcoral', 
                                           'gold', 'plum', 'orange'])
        ax.set_title(f'温度 = {temp}', fontsize=14)
        ax.set_ylabel('概率')
        ax.set_ylim(0, 1)
        
        # 添加数值标签
        for bar, prob in zip(bars, probs):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                   f'{prob:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('温度参数对概率分布的影响\n温度越低，概率分布越"尖锐"（更确定）', fontsize=14)
    plt.tight_layout()
    plt.savefig('04_语言模型/images/temperature_effect.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("已生成: temperature_effect.png")


def plot_autoregressive_generation():
    """
    绘制自回归生成过程图
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    steps = [
        ("输入: 空", "预测: 我"),
        ("输入: 我", "预测: 爱"),
        ("输入: 我 爱", "预测: 学习"),
        ("输入: 我 爱 学习", "预测: <END>"),
    ]
    
    y_positions = range(len(steps))
    
    for i, (input_text, output_text) in enumerate(steps):
        # 绘制输入框
        ax.text(0.1, i, input_text, fontsize=12, ha='left', va='center',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # 绘制箭头
        ax.annotate('', xy=(0.6, i), xytext=(0.4, i),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2))
        
        # 绘制输出框
        ax.text(0.7, i, output_text, fontsize=12, ha='left', va='center',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(steps) - 0.5)
    ax.set_title('自回归生成过程\n每次生成一个词，然后加入输入继续生成', fontsize=14)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('04_语言模型/images/autoregressive_generation.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("已生成: autoregressive_generation.png")


def main():
    """主函数"""
    
    print("=" * 50)
    print("生成第4章图片")
    print("=" * 50)
    print()
    
    # 创建images目录（如果不存在）
    import os
    os.makedirs('04_语言模型/images', exist_ok=True)
    
    # 生成图片
    plot_tokenization_comparison()
    plot_embedding_visualization()
    plot_temperature_effect()
    plot_autoregressive_generation()
    
    print()
    print("=" * 50)
    print("图片生成完成！")
    print("=" * 50)


if __name__ == "__main__":
    main()
