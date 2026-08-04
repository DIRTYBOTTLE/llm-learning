"""
4.5 GPT 生成过程演示

这个文件演示：
1. 自回归生成
2. 贪心解码
3. 采样解码
4. Top-K 和 Top-P 采样
5. 温度参数的影响
"""

import torch
import torch.nn.functional as F
import numpy as np


def demonstrate_autoregressive_generation():
    """
    演示自回归生成过程
    """
    print("=" * 60)
    print("自回归生成过程")
    print("=" * 60)
    print()
    
    print("自回归生成：每次生成一个词，然后加入输入继续生成")
    print("-" * 40)
    print()
    
    # 模拟一个简单的语言模型
    # 输入 -> 输出（简化版）
    transitions = {
        "": {"我": 0.8, "你": 0.15, "他": 0.05},
        "我": {"爱": 0.6, "喜欢": 0.3, "想": 0.1},
        "我 爱": {"学习": 0.5, "你": 0.3, "编程": 0.2},
        "我 爱 学习": {"<END>": 1.0}
    }
    
    print("模拟生成过程：")
    print()
    
    current_text = ""
    
    for step in range(5):
        print(f"步骤 {step + 1}:")
        print(f"  当前输入: '{current_text}'")
        
        # 获取下一个词的概率分布
        if current_text in transitions:
            probs = transitions[current_text]
        else:
            probs = {"<END>": 1.0}
        
        # 选择概率最大的词（贪心解码）
        next_word = max(probs, key=probs.get)
        
        print(f"  预测概率: {probs}")
        print(f"  选择的词: '{next_word}'")
        print()
        
        # 如果是结束符，停止
        if next_word == "<END>":
            print(f"生成完成！最终文本: '{current_text}'")
            break
        
        # 更新当前文本
        if current_text:
            current_text = current_text + " " + next_word
        else:
            current_text = next_word
    
    return current_text


def demonstrate_greedy_decoding():
    """
    演示贪心解码
    """
    print("=" * 60)
    print("贪心解码")
    print("=" * 60)
    print()
    
    print("贪心解码：每次都选择概率最大的词")
    print("-" * 40)
    print()
    
    # 模拟模型输出
    vocab = ["我", "爱", "学习", "吃", "饭", "你"]
    logits = torch.tensor([0.1, 0.5, 0.3, 0.05, 0.03, 0.02])
    
    print("模型输出的概率分布：")
    for word, logit in zip(vocab, logits):
        prob = torch.softmax(logits, dim=0)[vocab.index(word)]
        bar = "█" * int(prob * 50)
        print(f"  {word}: {prob:.4f} {bar}")
    print()
    
    # 贪心解码：选择概率最大的
    probs = torch.softmax(logits, dim=0)
    greedy_idx = torch.argmax(probs).item()
    greedy_word = vocab[greedy_idx]
    
    print(f"贪心解码结果: '{greedy_word}'")
    print(f"  -> 总是选择概率最大的词")
    print()
    
    print("优点：")
    print("  - 简单，确定性")
    print("  - 每次都选最好的")
    print()
    
    print("缺点：")
    print("  - 生成的文本可能重复、无趣")
    print("  - 缺乏多样性")
    print()
    
    return greedy_word


def demonstrate_sampling():
    """
    演示采样解码
    """
    print("=" * 60)
    print("采样解码")
    print("=" * 60)
    print()
    
    print("采样解码：根据概率分布随机采样")
    print("-" * 40)
    print()
    
    # 模拟模型输出
    vocab = ["我", "爱", "学习", "吃", "饭", "你"]
    logits = torch.tensor([0.1, 0.5, 0.3, 0.05, 0.03, 0.02])
    
    print("模型输出的概率分布：")
    probs = torch.softmax(logits, dim=0)
    for word, prob in zip(vocab, probs):
        bar = "█" * int(prob * 50)
        print(f"  {word}: {prob:.4f} {bar}")
    print()
    
    # 采样解码
    print("采样解码演示（采样5次）：")
    for i in range(5):
        sampled_idx = torch.multinomial(probs, num_samples=1).item()
        sampled_word = vocab[sampled_idx]
        print(f"  第{i+1}次采样: '{sampled_word}'")
    print()
    
    print("优点：")
    print("  - 生成的文本更多样、更有趣")
    print("  - 每次生成可能不同")
    print()
    
    print("缺点：")
    print("  - 可能生成不合理的词")
    print("  - 随机性太大")
    print()


def demonstrate_temperature():
    """
    演示温度参数的影响
    """
    print("=" * 60)
    print("温度参数的影响")
    print("=" * 60)
    print()
    
    print("温度（Temperature）控制生成的随机性")
    print("-" * 40)
    print()
    
    # 模拟模型输出
    vocab = ["我", "爱", "学习", "吃", "饭", "你"]
    logits = torch.tensor([0.1, 0.5, 0.3, 0.05, 0.03, 0.02])
    
    print("原始 logits:", logits.tolist())
    print()
    
    # 不同温度
    temperatures = [0.5, 1.0, 2.0]
    
    for temp in temperatures:
        # 温度缩放
        scaled_logits = logits / temp
        probs = torch.softmax(scaled_logits, dim=0)
        
        print(f"温度 = {temp}:")
        for word, prob in zip(vocab, probs):
            bar = "█" * int(prob * 50)
            print(f"  {word}: {prob:.4f} {bar}")
        print()
    
    print("温度的影响：")
    print("  - 温度低（0.5）：概率分布更\"尖锐\"，更确定")
    print("  - 温度高（2.0）：概率分布更\"平坦\"，更随机")
    print("  - 温度 = 1.0：正常")
    print()
    
    print("使用建议：")
    print("  - 需要确定性：低温度（0.3-0.5）")
    print("  - 需要多样性：高温度（1.2-1.5）")
    print("  - 一般情况：温度 = 0.7-1.0")
    print()


def demonstrate_top_k_sampling():
    """
    演示 Top-K 采样
    """
    print("=" * 60)
    print("Top-K 采样")
    print("=" * 60)
    print()
    
    print("Top-K 采样：只从概率最大的 K 个词中采样")
    print("-" * 40)
    print()
    
    # 模拟模型输出
    vocab = ["我", "爱", "学习", "吃", "饭", "你", "他", "她", "它", "们"]
    logits = torch.tensor([0.1, 0.5, 0.3, 0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.002])
    
    print("原始概率分布：")
    probs = torch.softmax(logits, dim=0)
    for word, prob in zip(vocab, probs):
        print(f"  {word}: {prob:.4f}")
    print()
    
    # Top-K 采样
    k_values = [3, 5, 10]
    
    for k in k_values:
        print(f"Top-K = {k}：")
        
        # 取最大的 K 个
        top_k_probs, top_k_indices = torch.topk(probs, k)
        
        print("  保留的词：")
        for idx, prob in zip(top_k_indices, top_k_probs):
            print(f"    {vocab[idx]}: {prob:.4f}")
        
        # 重新归一化
        top_k_probs = top_k_probs / top_k_probs.sum()
        
        # 采样
        sampled_idx = torch.multinomial(top_k_probs, num_samples=1).item()
        sampled_word = vocab[top_k_indices[sampled_idx]]
        
        print(f"  采样结果: '{sampled_word}'")
        print()
    
    print("Top-K 的作用：")
    print("  - 过滤掉概率很低的词")
    print("  - 避免生成不合理的词")
    print("  - K 越小，越保守；K 越大，越多样")
    print()


def demonstrate_top_p_sampling():
    """
    演示 Top-P 采样（Nucleus Sampling）
    """
    print("=" * 60)
    print("Top-P 采样（Nucleus Sampling）")
    print("=" * 60)
    print()
    
    print("Top-P 采样：从累积概率达到 P 的词中采样")
    print("-" * 40)
    print()
    
    # 模拟模型输出
    vocab = ["我", "爱", "学习", "吃", "饭", "你", "他", "她", "它", "们"]
    logits = torch.tensor([0.1, 0.5, 0.3, 0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.002])
    
    probs = torch.softmax(logits, dim=0)
    
    # 排序
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    
    print("按概率排序：")
    cumulative_prob = 0
    for idx, prob in zip(sorted_indices, sorted_probs):
        cumulative_prob += prob.item()
        print(f"  {vocab[idx]}: {prob:.4f} (累积: {cumulative_prob:.4f})")
    print()
    
    # Top-P 采样
    p_values = [0.5, 0.8, 0.95]
    
    for p in p_values:
        print(f"Top-P = {p}：")
        
        # 计算累积概率
        cumulative_probs = torch.cumsum(sorted_probs, dim=0)
        
        # 找到累积概率超过 P 的位置
        mask = cumulative_probs - sorted_probs > p
        
        # 过滤
        filtered_probs = sorted_probs.clone()
        filtered_probs[mask] = 0
        
        # 重新归一化
        filtered_probs = filtered_probs / filtered_probs.sum()
        
        # 显示保留的词
        print("  保留的词：")
        for idx, prob, keep in zip(sorted_indices, sorted_probs, ~mask):
            if keep:
                print(f"    {vocab[idx]}: {prob:.4f}")
        
        # 采样
        non_zero_indices = torch.where(filtered_probs > 0)[0]
        if len(non_zero_indices) > 0:
            sampled_pos = torch.multinomial(filtered_probs[non_zero_indices], num_samples=1).item()
            sampled_idx = sorted_indices[non_zero_indices[sampled_pos]]
            sampled_word = vocab[sampled_idx]
            print(f"  采样结果: '{sampled_word}'")
        print()
    
    print("Top-P 的优势：")
    print("  - 自适应：根据概率分布动态调整")
    print("  - 如果概率分布尖锐，只保留少数词")
    print("  - 如果概率分布平坦，保留更多词")
    print()


def demonstrate_generation_comparison():
    """
    演示不同生成方法的对比
    """
    print("=" * 60)
    print("不同生成方法的对比")
    print("=" * 60)
    print()
    
    # 模拟模型输出
    vocab = ["我", "爱", "学习", "吃", "饭", "你"]
    logits = torch.tensor([0.1, 0.5, 0.3, 0.05, 0.03, 0.02])
    probs = torch.softmax(logits, dim=0)
    
    print("原始概率分布：")
    for word, prob in zip(vocab, probs):
        print(f"  {word}: {prob:.4f}")
    print()
    
    print("不同方法的结果（模拟）：")
    print("-" * 40)
    
    # 贪心解码
    greedy_idx = torch.argmax(probs).item()
    print(f"贪心解码: '{vocab[greedy_idx]}'（总是选最大的）")
    
    # 采样解码
    sampled_results = []
    for _ in range(5):
        idx = torch.multinomial(probs, num_samples=1).item()
        sampled_results.append(vocab[idx])
    print(f"采样解码: {sampled_results}（每次可能不同）")
    
    # Top-K 采样
    top_k_probs, top_k_indices = torch.topk(probs, 3)
    top_k_probs = top_k_probs / top_k_probs.sum()
    top_k_results = []
    for _ in range(5):
        idx = torch.multinomial(top_k_probs, num_samples=1).item()
        top_k_results.append(vocab[top_k_indices[idx]])
    print(f"Top-K=3:  {top_k_results}（只从最大的3个中选）")
    
    print()
    print("选择建议：")
    print("  - 代码生成：贪心解码（确定性）")
    print("  - 创意写作：采样 + 高温度（多样性）")
    print("  - 一般对话：Top-P=0.9 + 温度=0.7（平衡）")
    print()


def main():
    """主函数"""
    
    print("\n" + "=" * 60)
    print("第4章 第5节：GPT 生成过程演示")
    print("=" * 60 + "\n")
    
    # 1. 自回归生成
    demonstrate_autoregressive_generation()
    
    print("-" * 60 + "\n")
    
    # 2. 贪心解码
    demonstrate_greedy_decoding()
    
    print("-" * 60 + "\n")
    
    # 3. 采样解码
    demonstrate_sampling()
    
    print("-" * 60 + "\n")
    
    # 4. 温度参数
    demonstrate_temperature()
    
    print("-" * 60 + "\n")
    
    # 5. Top-K 采样
    demonstrate_top_k_sampling()
    
    print("-" * 60 + "\n")
    
    # 6. Top-P 采样
    demonstrate_top_p_sampling()
    
    print("-" * 60 + "\n")
    
    # 7. 对比
    demonstrate_generation_comparison()
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
