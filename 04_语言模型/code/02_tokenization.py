"""
4.2 Tokenization（分词）演示

这个文件演示三种分词方法：
1. 字符级分词
2. 词级分词
3. 子词级分词（BPE）
"""

import re
from collections import Counter, defaultdict


def character_tokenization(text):
    """
    字符级分词：把每个字符当作一个 token
    """
    print("字符级分词：")
    print("-" * 40)
    
    tokens = list(text)
    
    print(f"输入: '{text}'")
    print(f"输出: {tokens}")
    print(f"Token 数量: {len(tokens)}")
    print()
    
    return tokens


def word_tokenization(text):
    """
    词级分词：把每个词当作一个 token
    """
    print("词级分词：")
    print("-" * 40)
    
    # 简单的分词：按空格分割
    tokens = text.split()
    
    print(f"输入: '{text}'")
    print(f"输出: {tokens}")
    print(f"Token 数量: {len(tokens)}")
    print()
    
    return tokens


def simple_bpe(text, num_merges=5):
    """
    简化的 BPE 算法演示
    """
    print("BPE 分词（简化版）：")
    print("-" * 40)
    
    # 把文本分成字符
    words = text.split()
    
    # 统计词频
    word_freq = Counter(words)
    
    print(f"输入文本: '{text}'")
    print(f"词频统计: {dict(word_freq)}")
    print()
    
    # 把每个词分成字符
    vocab = {}
    for word, freq in word_freq.items():
        # 在每个字符后面加一个结束符 '_'
        chars = list(word) + ['_']
        vocab[tuple(chars)] = freq
    
    print("初始词汇表（字符级）：")
    for chars, freq in vocab.items():
        print(f"  {' '.join(chars)}: {freq}次")
    print()
    
    # 学习合并规则
    merges = []
    
    for i in range(num_merges):
        # 统计所有相邻 pair 的频率
        pairs = Counter()
        for chars, freq in vocab.items():
            for j in range(len(chars) - 1):
                pairs[(chars[j], chars[j+1])] += freq
        
        if not pairs:
            break
        
        # 找出最频繁的 pair
        best_pair = max(pairs, key=pairs.get)
        merges.append(best_pair)
        
        print(f"第 {i+1} 次合并：")
        print(f"  最频繁的 pair: {best_pair}（出现 {pairs[best_pair]} 次）")
        
        # 合并这个 pair
        new_vocab = {}
        for chars, freq in vocab.items():
            new_chars = []
            j = 0
            while j < len(chars):
                if j < len(chars) - 1 and (chars[j], chars[j+1]) == best_pair:
                    new_chars.append(chars[j] + chars[j+1])
                    j += 2
                else:
                    new_chars.append(chars[j])
                    j += 1
            new_vocab[tuple(new_chars)] = freq
        
        vocab = new_vocab
        
        print(f"  合并后的词汇表：")
        for chars, freq in vocab.items():
            print(f"    {' '.join(chars)}: {freq}次")
        print()
    
    return merges, vocab


def demonstrate_token_to_index():
    """
    演示 token 到索引的转换
    """
    print("Token 到索引的转换：")
    print("-" * 40)
    
    # 假设的词表
    vocab = {
        "我": 0,
        "爱": 1,
        "学习": 2,
        "深度": 3,
        "学习_": 4,  # BPE 可能产生这样的 token
        "<UNK>": 5,
        "<PAD>": 6,
        "<START>": 7,
        "<END>": 8
    }
    
    print("词表：")
    for token, idx in vocab.items():
        print(f"  {token}: {idx}")
    print()
    
    # 编码过程
    text = "我爱学习"
    tokens = ["我", "爱", "学习"]
    
    print(f"编码过程：")
    print(f"  原始文本: '{text}'")
    print(f"  分词结果: {tokens}")
    
    # 转换成索引
    indices = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    print(f"  索引序列: {indices}")
    print()
    
    # 解码过程
    print(f"解码过程：")
    print(f"  索引序列: {indices}")
    
    idx_to_token = {v: k for k, v in vocab.items()}
    decoded_tokens = [idx_to_token[idx] for idx in indices]
    decoded_text = "".join(decoded_tokens)
    
    print(f"  Token 序列: {decoded_tokens}")
    print(f"  解码文本: '{decoded_text}'")
    print()


def demonstrate_oov_problem():
    """
    演示 OOV（Out-of-Vocabulary）问题
    """
    print("OOV 问题演示：")
    print("-" * 40)
    
    # 假设的词表
    vocab = {"我": 0, "爱": 1, "学习": 2, "深度": 3}
    
    print(f"词表: {list(vocab.keys())}")
    print()
    
    # 测试句子
    test_sentences = [
        "我爱学习",
        "我爱深度学习",
        "我喜欢学习"  # "喜欢"不在词表中
    ]
    
    for sentence in test_sentences:
        tokens = sentence  # 简化处理
        print(f"句子: '{sentence}'")
        
        # 检查是否有 OOV
        oov_tokens = [t for t in tokens if t not in vocab]
        
        if oov_tokens:
            print(f"  ⚠️ 发现 OOV 词: {oov_tokens}")
        else:
            print(f"  ✓ 所有词都在词表中")
        print()
    
    print("BPE 的优势：通过子词组合，可以处理未知词！")


def main():
    """主函数"""
    
    print("\n" + "=" * 60)
    print("第4章 第2节：Tokenization（分词）演示")
    print("=" * 60 + "\n")
    
    # 测试文本
    text = "我爱学习深度学习"
    
    # 1. 字符级分词
    character_tokenization(text)
    
    print("-" * 60 + "\n")
    
    # 2. 词级分词
    word_tokenization(text)
    
    print("-" * 60 + "\n")
    
    # 3. BPE 分词
    simple_bpe(text, num_merges=3)
    
    print("-" * 60 + "\n")
    
    # 4. Token 到索引的转换
    demonstrate_token_to_index()
    
    print("-" * 60 + "\n")
    
    # 5. OOV 问题演示
    demonstrate_oov_problem()
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
