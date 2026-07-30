"""
概率论与信息论演示
运行方式：python 00_数学基础/code/probability_and_entropy.py
依赖：pip install numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def basic_probability():
    """基础概率"""
    print("=" * 50)
    print("1. 基础概率")
    print("=" * 50)

    # 模拟抛硬币
    np.random.seed(42)
    n_flips = 10000
    flips = np.random.choice(['正面', '反面'], size=n_flips)

    p_heads = np.sum(flips == '正面') / n_flips
    p_tails = np.sum(flips == '反面') / n_flips

    print(f"模拟抛硬币 {n_flips} 次：")
    print(f"  正面次数: {np.sum(flips == '正面')}")
    print(f"  反面次数: {np.sum(flips == '反面')}")
    print(f"  P(正面) = {p_heads:.4f} (理论值: 0.5)")
    print(f"  P(反面) = {p_tails:.4f} (理论值: 0.5)")

    # 模拟掷骰子
    n_rolls = 6000
    rolls = np.random.randint(1, 7, size=n_rolls)

    print(f"\n模拟掷骰子 {n_rolls} 次：")
    for i in range(1, 7):
        count = np.sum(rolls == i)
        print(f"  出现 {i}: {count} 次, P = {count/n_rolls:.4f} (理论值: 1/6 ≈ 0.1667)")


def conditional_probability():
    """条件概率与贝叶斯定理"""
    print("\n" + "=" * 50)
    print("2. 条件概率与贝叶斯定理")
    print("=" * 50)

    # 经典例子：医学检测
    # 疾病患病率: 1%
    # 检测准确率: 99%（有病检测出阳性）
    # 假阳性率: 5%（没病检测出阳性）

    p_disease = 0.01  # 先验概率
    p_positive_given_disease = 0.99  # 似然
    p_positive_given_no_disease = 0.05  # 假阳性率

    # 计算 P(阳性)
    p_positive = (p_positive_given_disease * p_disease +
                  p_positive_given_no_disease * (1 - p_disease))

    # 贝叶斯定理：P(有病|阳性)
    p_disease_given_positive = (p_positive_given_disease * p_disease) / p_positive

    print("医学检测问题：")
    print(f"  疾病患病率 P(有病) = {p_disease}")
    print(f"  检测准确率 P(阳性|有病) = {p_positive_given_disease}")
    print(f"  假阳性率 P(阳性|没病) = {p_positive_given_no_disease}")
    print(f"\n  P(阳性) = {p_positive:.4f}")
    print(f"  P(有病|阳性) = {p_disease_given_positive:.4f}")
    print(f"\n  结论：即使检测阳性，真正患病的概率只有 {p_disease_given_positive*100:.1f}%！")
    print(f"  这是因为疾病本身很罕见（先验概率低）")


def probability_distributions():
    """概率分布"""
    print("\n" + "=" * 50)
    print("3. 常见概率分布")
    print("=" * 50)

    # 伯努利分布
    p = 0.7
    print(f"伯努利分布 (p={p}):")
    print(f"  P(X=1) = {p}")
    print(f"  P(X=0) = {1-p}")
    print(f"  期望 E(X) = {p}")
    print(f"  方差 Var(X) = {p*(1-p):.4f}")

    # 二项分布
    n, p = 10, 0.5
    print(f"\n二项分布 (n={n}, p={p}):")
    k = np.arange(n + 1)
    binom_pmf = np.array([np.math.comb(n, i) * p**i * (1-p)**(n-i) for i in k])
    print(f"  期望 E(X) = {n*p}")
    print(f"  方差 Var(X) = {n*p*(1-p):.4f}")

    # 正态分布
    mu, sigma = 0, 1
    print(f"\n正态分布 (μ={mu}, σ={sigma}):")
    print(f"  期望 E(X) = {mu}")
    print(f"  方差 Var(X) = {sigma**2}")
    print(f"  P(-1 < X < 1) ≈ 0.6827 (68-95-99.7法则)")
    print(f"  P(-2 < X < 2) ≈ 0.9545")
    print(f"  P(-3 < X < 3) ≈ 0.9973")


def entropy_and_cross_entropy():
    """熵与交叉熵"""
    print("\n" + "=" * 50)
    print("4. 熵与交叉熵")
    print("=" * 50)

    def entropy(probs):
        """计算熵"""
        probs = np.array(probs)
        probs = probs[probs > 0]  # 避免 log(0)
        return -np.sum(probs * np.log2(probs))

    def cross_entropy(true_probs, pred_probs):
        """计算交叉熵"""
        true_probs = np.array(true_probs)
        pred_probs = np.array(pred_probs)
        pred_probs = pred_probs[pred_probs > 0]
        true_probs = true_probs[pred_probs > 0]
        return -np.sum(true_probs * np.log2(pred_probs))

    # 公平硬币
    fair_coin = [0.5, 0.5]
    print(f"公平硬币：P = {fair_coin}")
    print(f"  熵 H = {entropy(fair_coin):.4f} bits")

    # 不公平硬币
    unfair_coin = [0.9, 0.1]
    print(f"\n不公平硬币：P = {unfair_coin}")
    print(f"  熵 H = {entropy(unfair_coin):.4f} bits")
    print(f"  （不确定性更低，因为结果更可预测）")

    # 交叉熵示例
    true_dist = [0.5, 0.5]  # 真实分布：公平硬币
    pred_dist1 = [0.5, 0.5]  # 完美预测
    pred_dist2 = [0.7, 0.3]  # 有偏差的预测
    pred_dist3 = [0.9, 0.1]  # 很有偏差的预测

    print(f"\n交叉熵（真实分布：{true_dist}）：")
    print(f"  预测 {pred_dist1}: H = {cross_entropy(true_dist, pred_dist1):.4f} (完美)")
    print(f"  预测 {pred_dist2}: H = {cross_entropy(true_dist, pred_dist2):.4f}")
    print(f"  预测 {pred_dist3}: H = {cross_entropy(true_dist, pred_dist3):.4f} (偏差大)")

    # 交叉熵作为损失函数
    print("\n在语言模型中的应用：")
    print("  真实下一个词：'猫' (one-hot: [1, 0, 0])")
    print("  模型预测：[0.7, 0.2, 0.1]")
    print(f"  交叉熵损失 = {-np.log2(0.7):.4f}")
    print("  目标：调整模型参数，使损失最小化")


def kl_divergence():
    """KL 散度"""
    print("\n" + "=" * 50)
    print("5. KL 散度")
    print("=" * 50)

    def kl_divergence(p, q):
        """计算 KL(P||Q)"""
        p = np.array(p)
        q = np.array(q)
        # 避免除以0
        mask = (p > 0) & (q > 0)
        p, q = p[mask], q[mask]
        return np.sum(p * np.log(p / q))

    # 示例
    p = [0.5, 0.3, 0.2]  # 真实分布
    q1 = [0.5, 0.3, 0.2]  # 相同分布
    q2 = [0.4, 0.4, 0.2]  # 略有不同
    q3 = [0.1, 0.1, 0.8]  # 差异很大

    print(f"真实分布 P = {p}")
    print(f"\nKL 散度 KL(P||Q)：")
    print(f"  Q = {q1}: KL = {kl_divergence(p, q1):.6f} (相同分布，KL=0)")
    print(f"  Q = {q2}: KL = {kl_divergence(p, q2):.6f}")
    print(f"  Q = {q3}: KL = {kl_divergence(p, q3):.6f} (差异大，KL大)")

    print("\nKL 散度的性质：")
    print("  1. KL(P||Q) ≥ 0")
    print("  2. KL(P||Q) = 0 当且仅当 P = Q")
    print("  3. 不对称：KL(P||Q) ≠ KL(Q||P)")


def maximum_likelihood():
    """极大似然估计"""
    print("\n" + "=" * 50)
    print("6. 极大似然估计")
    print("=" * 50)

    # 抛硬币实验
    np.random.seed(42)
    true_p = 0.7  # 真实概率
    n_trials = 100
    data = np.random.binomial(1, true_p, size=n_trials)

    n_heads = np.sum(data == 1)
    n_tails = np.sum(data == 0)

    # 极大似然估计
    mle_p = n_heads / n_trials

    print(f"抛硬币实验：{n_trials} 次")
    print(f"  正面次数: {n_heads}")
    print(f"  反面次数: {n_tails}")
    print(f"\n极大似然估计：")
    print(f"  真实概率: {true_p}")
    print(f"  估计概率: {mle_p:.4f}")
    print(f"  估计误差: {abs(mle_p - true_p):.4f}")

    # 不同样本量下的估计
    print("\n不同样本量下的估计精度：")
    for n in [10, 50, 100, 500, 1000]:
        sample = np.random.binomial(1, true_p, size=n)
        estimate = np.sum(sample) / n
        print(f"  n={n:4d}: 估计={estimate:.4f}, 误差={abs(estimate-true_p):.4f}")


def visualize_distributions():
    """可视化概率分布"""
    print("\n" + "=" * 50)
    print("7. 概率分布可视化")
    print("=" * 50)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. 伯努利分布
    ax1 = axes[0, 0]
    p = 0.7
    ax1.bar([0, 1], [1-p, p], color=['skyblue', 'salmon'])
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['失败 (0)', '成功 (1)'])
    ax1.set_ylabel('概率')
    ax1.set_title(f'伯努利分布 (p={p})')
    ax1.set_ylim(0, 1)

    # 2. 二项分布
    ax2 = axes[0, 1]
    n, p = 20, 0.5
    k = np.arange(n + 1)
    binom_pmf = np.array([np.math.comb(n, i) * p**i * (1-p)**(n-i) for i in k])
    ax2.bar(k, binom_pmf, color='steelblue')
    ax2.set_xlabel('成功次数')
    ax2.set_ylabel('概率')
    ax2.set_title(f'二项分布 (n={n}, p={p})')

    # 3. 正态分布
    ax3 = axes[1, 0]
    x = np.linspace(-4, 4, 100)
    for mu, sigma, label in [(0, 1, 'μ=0, σ=1'), (0, 0.5, 'μ=0, σ=0.5'), (0, 2, 'μ=0, σ=2')]:
        y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        ax3.plot(x, y, label=label, linewidth=2)
    ax3.set_xlabel('x')
    ax3.set_ylabel('概率密度')
    ax3.set_title('正态分布')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 熵与概率的关系
    ax4 = axes[1, 1]
    p_values = np.linspace(0.01, 0.99, 100)
    entropies = [-p * np.log2(p) - (1-p) * np.log2(1-p) for p in p_values]
    ax4.plot(p_values, entropies, 'b-', linewidth=2)
    ax4.axvline(x=0.5, color='r', linestyle='--', label='最大熵点 (p=0.5)')
    ax4.set_xlabel('P(正面)')
    ax4.set_ylabel('熵 (bits)')
    ax4.set_title('硬币熵与概率的关系')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('00_数学基础/code/probability_distributions.png', dpi=150, bbox_inches='tight')
    print("可视化已保存到 probability_distributions.png")
    plt.show()


def language_model_connection():
    """与语言模型的联系"""
    print("\n" + "=" * 50)
    print("8. 与语言模型的联系")
    print("=" * 50)

    # 模拟一个简单的语言模型
    vocab = ['我', '喜欢', '猫', '狗', '吃', '鱼']
    true_probs = [0.2, 0.15, 0.25, 0.1, 0.15, 0.15]  # 真实分布
    model_probs = [0.18, 0.17, 0.22, 0.12, 0.16, 0.15]  # 模型预测

    print("词汇表：", vocab)
    print(f"真实分布：{dict(zip(vocab, true_probs))}")
    print(f"模型预测：{dict(zip(vocab, model_probs))}")

    # 计算交叉熵损失
    def cross_entropy(true_probs, pred_probs):
        true_probs = np.array(true_probs)
        pred_probs = np.array(pred_probs)
        return -np.sum(true_probs * np.log2(pred_probs))

    loss = cross_entropy(true_probs, model_probs)
    print(f"\n交叉熵损失 = {loss:.4f} bits")
    print("\n训练目标：调整模型参数，使交叉熵损失最小化")
    print("当模型预测 = 真实分布时，损失达到最小值")


if __name__ == "__main__":
    print("概率论与信息论演示\n")
    print("本演示将展示大模型中核心的概率论概念\n")

    basic_probability()
    conditional_probability()
    probability_distributions()
    entropy_and_cross_entropy()
    kl_divergence()
    maximum_likelihood()
    visualize_distributions()
    language_model_connection()

    print("\n" + "=" * 50)
    print("演示完成！")
    print("=" * 50)
    print("\n关键要点：")
    print("1. 概率描述事件发生的可能性")
    print("2. 贝叶斯定理用于根据证据更新信念")
    print("3. 交叉熵是语言模型的损失函数")
    print("4. KL 散度衡量两个分布的差异")
    print("5. 语言模型训练 = 极大似然估计 = 最小化交叉熵")
