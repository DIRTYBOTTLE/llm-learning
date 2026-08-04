"""
4.1 什么是语言模型 - 概念演示

这个文件演示语言模型的核心概念：
1. 语言模型的任务：预测下一个词
2. 自回归生成的过程
3. 概率的计算
"""

import torch
import torch.nn.functional as F


def demonstrate_prediction_task():
    """
    演示语言模型的核心任务：预测下一个词
    """
    print("=" * 60)
    print("语言模型的核心任务：预测下一个词")
    print("=" * 60)
    print()
    
    # 假设我们有一个简单的词表
    vocab = {"我": 0, "爱": 1, "学习": 2, "吃": 3, "饭": 4, "<END>": 5}
    idx_to_word = {v: k for k, v in vocab.items()}
    
    print("词表：")
    for word, idx in vocab.items():
        print(f"  {word} -> {idx}")
    print()
    
    # 模拟语言模型的输出（logits）
    # 假设输入是"我 爱"，模型预测下一个词
    print("场景：输入'我 爱'，预测下一个词")
    print("-" * 40)
    
    # 模型输出的 logits（未归一化的分数）
    logits = torch.tensor([0.1, 0.2, 2.5, 0.3, 0.1, 0.05])
    
    # 转换成概率
    probs = F.softmax(logits, dim=0)
    
    print("模型输出的概率分布：")
    for idx, (word, prob) in enumerate(zip(vocab.keys(), probs)):
        bar = "█" * int(prob * 50)
        print(f"  {word:6s}: {prob:.4f} {bar}")
    print()
    
    # 选择概率最大的词
    predicted_idx = torch.argmax(probs).item()
    predicted_word = idx_to_word[predicted_idx]
    
    print(f"预测结果：'{predicted_word}'（概率最高）")
    print()
    
    return predicted_word


def demonstrate_autoregressive():
    """
    演示自回归生成过程
    """
    print("=" * 60)
    print("自回归生成过程")
    print("=" * 60)
    print()
    
    # 模拟一个简单的语言模型
    # 输入 -> 输出（简化版）
    transitions = {
        "": "我",
        "我": "爱",
        "我 爱": "学习",
        "我 爱 学习": "<END>"
    }
    
    print("自回归生成演示：")
    print("-" * 40)
    
    current_text = ""
    
    for step in range(5):
        # 获取下一个词
        if current_text in transitions:
            next_word = transitions[current_text]
        else:
            next_word = "<END>"
        
        print(f"步骤 {step + 1}:")
        print(f"  输入: '{current_text}'")
        print(f"  预测: '{next_word}'")
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


def demonstrate_probability_chain():
    """
    演示概率链式法则
    """
    print("=" * 60)
    print("概率链式法则")
    print("=" * 60)
    print()
    
    print("语言模型的概率公式：")
    print("P(w1, w2, w3) = P(w1) × P(w2|w1) × P(w3|w1,w2)")
    print()
    
    # 假设的概率
    p_我 = 0.15
    p_爱_我 = 0.40
    p_学习_我爱 = 0.60
    
    print("例子：计算 P('我 爱 学习')")
    print("-" * 40)
    print(f"  P('我') = {p_我}")
    print(f"  P('爱' | '我') = {p_爱_我}")
    print(f"  P('学习' | '我 爱') = {p_学习_我爱}")
    print()
    
    # 计算联合概率
    joint_prob = p_我 * p_爱_我 * p_学习_我爱
    
    print(f"P('我 爱 学习') = {p_我} × {p_爱_我} × {p_学习_我爱}")
    print(f"               = {joint_prob:.4f}")
    print()
    print(f"所以'我爱学习'这个句子的概率是 {joint_prob:.2%}")
    print()
    
    return joint_prob


def demonstrate_training_objective():
    """
    演示训练目标
    """
    print("=" * 60)
    print("训练目标：最大化下一个词的概率")
    print("=" * 60)
    print()
    
    print("训练时，我们有真实的句子：'我 爱 学习'")
    print()
    
    # 训练样本
    training_samples = [
        ("", "我"),
        ("我", "爱"),
        ("我 爱", "学习")
    ]
    
    print("构造训练样本：")
    print("-" * 40)
    for input_text, target in training_samples:
        print(f"  输入: '{input_text}' -> 目标: '{target}'")
    print()
    
    print("训练目标：")
    print("  最大化 P(目标 | 输入)")
    print("  等价于最小化 -log P(目标 | 输入)")
    print()
    
    # 模拟训练过程
    print("模拟训练过程：")
    print("-" * 40)
    
    # 初始概率（随机）
    probs_initial = [0.1, 0.2, 0.3]
    # 训练后的概率（应该更接近1）
    probs_trained = [0.1, 0.4, 0.7]
    
    print("训练前（随机概率）：")
    for i, (input_text, target) in enumerate(training_samples):
        print(f"  P('{target}' | '{input_text}') = {probs_initial[i]:.2f}")
    print()
    
    print("训练后（学习到的概率）：")
    for i, (input_text, target) in enumerate(training_samples):
        print(f"  P('{target}' | '{input_text}') = {probs_trained[i]:.2f}")
    print()
    
    print("训练后，模型更准确地预测下一个词了！")


def main():
    """主函数"""
    
    print("\n" + "=" * 60)
    print("第4章 第1节：什么是语言模型 - 概念演示")
    print("=" * 60 + "\n")
    
    # 1. 演示预测任务
    demonstrate_prediction_task()
    
    print("\n" + "-" * 60 + "\n")
    
    # 2. 演示自回归生成
    demonstrate_autoregressive()
    
    print("\n" + "-" * 60 + "\n")
    
    # 3. 演示概率链式法则
    demonstrate_probability_chain()
    
    print("\n" + "-" * 60 + "\n")
    
    # 4. 演示训练目标
    demonstrate_training_objective()
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
