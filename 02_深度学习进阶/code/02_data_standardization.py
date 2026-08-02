"""
第2章 2.2 数据标准化
code/02_data_standardization.py

功能：演示数据标准化的效果
对应教程：02_数据标准化.md
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


def create_simple_model():
    """创建简单的线性模型"""

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 1)

        def forward(self, x):
            return self.linear(x)

    return SimpleModel()


def train_model(model, X_train, y_train, X_val, y_val, epochs=200, lr=0.01):
    """训练模型"""

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # 训练
        model.train()
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        # 验证
        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val)
            val_loss = loss_fn(y_val_pred, y_val)
            val_losses.append(val_loss.item())

    return train_losses, val_losses


def demo_no_standardization():
    """演示没有标准化的情况"""
    print("=" * 60)
    print("1. 没有标准化的情况")
    print("=" * 60)

    # 生成数据：面积（50-200），房间数（1-5）
    torch.manual_seed(42)
    n_samples = 100

    # 特征：[面积, 房间数]
    area = torch.rand(n_samples, 1) * 150 + 50  # 50 ~ 200
    rooms = torch.rand(n_samples, 1) * 4 + 1    # 1 ~ 5
    X = torch.cat([area, rooms], dim=1)

    # 目标：房价 = 面积 * 2 + 房间数 * 30 + 噪声
    y = area * 2 + rooms * 30 + torch.randn(n_samples, 1) * 20

    # 划分训练集和验证集
    X_train, X_val = X[:80], X[80:]
    y_train, y_val = y[:80], y[80:]

    print(f"\n数据特征：")
    print(f"  面积：范围 [{X[:, 0].min():.0f}, {X[:, 0].max():.0f}]")
    print(f"  房间数：范围 [{X[:, 1].min():.0f}, {X[:, 1].max():.0f}]")

    # 训练
    model = create_simple_model()
    train_losses, val_losses = train_model(model, X_train, y_train, X_val, y_val, epochs=200, lr=0.0001)

    print(f"\n训练结果：")
    print(f"  最终训练损失：{train_losses[-1]:.2f}")
    print(f"  最终验证损失：{val_losses[-1]:.2f}")

    # 查看权重
    w1, w2 = model.linear.weight.data[0]
    print(f"\n学习到的权重：")
    print(f"  w1（面积）：{w1:.4f}")
    print(f"  w2（房间数）：{w2:.4f}")
    print(f"\n真实权重：w1=2, w2=30")
    print(f"→ 权重尺度差异大，训练不稳定")


def demo_standardization():
    """演示标准化后的效果"""
    print("\n" + "=" * 60)
    print("2. 标准化后的效果")
    print("=" * 60)

    # 生成数据
    torch.manual_seed(42)
    n_samples = 100

    area = torch.rand(n_samples, 1) * 150 + 50
    rooms = torch.rand(n_samples, 1) * 4 + 1
    X = torch.cat([area, rooms], dim=1)

    y = area * 2 + rooms * 30 + torch.randn(n_samples, 1) * 20

    # 标准化
    mean = X.mean(dim=0)
    std = X.std(dim=0)
    X_scaled = (X - mean) / std

    print(f"\n标准化前：")
    print(f"  面积：范围 [{X[:, 0].min():.0f}, {X[:, 0].max():.0f}]")
    print(f"  房间数：范围 [{X[:, 1].min():.0f}, {X[:, 1].max():.0f}]")

    print(f"\n标准化后：")
    print(f"  面积：范围 [{X_scaled[:, 0].min():.2f}, {X_scaled[:, 0].max():.2f}]")
    print(f"  房间数：范围 [{X_scaled[:, 1].min():.2f}, {X_scaled[:, 1].max():.2f}]")
    print(f"  面积均值：{X_scaled[:, 0].mean():.2f}，标准差：{X_scaled[:, 0].std():.2f}")
    print(f"  房间数均值：{X_scaled[:, 1].mean():.2f}，标准差：{X_scaled[:, 1].std():.2f}")

    # 划分训练集和验证集
    X_train, X_val = X_scaled[:80], X_scaled[80:]
    y_train, y_val = y[:80], y[80:]

    # 训练
    model = create_simple_model()
    train_losses, val_losses = train_model(model, X_train, y_train, X_val, y_val, epochs=200, lr=0.01)

    print(f"\n训练结果：")
    print(f"  最终训练损失：{train_losses[-1]:.2f}")
    print(f"  最终验证损失：{val_losses[-1]:.2f}")

    # 查看权重
    w1, w2 = model.linear.weight.data[0]
    print(f"\n学习到的权重（标准化后）：")
    print(f"  w1（面积）：{w1:.4f}")
    print(f"  w2（房间数）：{w2:.4f}")
    print(f"\n注意：权重尺度相似，训练更稳定")


def demo_comparison():
    """对比有无标准化的训练过程"""
    print("\n" + "=" * 60)
    print("3. 对比有无标准化的训练过程")
    print("=" * 60)

    # 生成数据
    torch.manual_seed(42)
    n_samples = 100

    area = torch.rand(n_samples, 1) * 150 + 50
    rooms = torch.rand(n_samples, 1) * 4 + 1
    X = torch.cat([area, rooms], dim=1)

    y = area * 2 + rooms * 30 + torch.randn(n_samples, 1) * 20

    # 划分训练集和验证集
    X_train, X_val = X[:80], X[80:]
    y_train, y_val = y[:80], y[80:]

    # 标准化
    mean = X_train.mean(dim=0)
    std = X_train.std(dim=0)
    X_train_scaled = (X_train - mean) / std
    X_val_scaled = (X_val - mean) / std

    print(f"\n对比结果：")
    print("-" * 50)
    print(f"{'方法':<15} {'训练损失':<12} {'验证损失':<12}")
    print("-" * 50)

    # 没有标准化
    torch.manual_seed(42)
    model1 = create_simple_model()
    train_l1, val_l1 = train_model(model1, X_train, y_train, X_val, y_val, epochs=200, lr=0.0001)
    print(f"{'没有标准化':<15} {train_l1[-1]:<12.2f} {val_l1[-1]:<12.2f}")

    # 标准化后
    torch.manual_seed(42)
    model2 = create_simple_model()
    train_l2, val_l2 = train_model(model2, X_train_scaled, y_train, X_val_scaled, y_val, epochs=200, lr=0.01)
    print(f"{'标准化后':<15} {train_l2[-1]:<12.2f} {val_l2[-1]:<12.2f}")

    print("-" * 50)
    print("\n→ 标准化后，训练损失更低，验证损失也更低")


def demo_standardization_vs_normalization():
    """对比标准化和归一化"""
    print("\n" + "=" * 60)
    print("4. 标准化 vs 归一化")
    print("=" * 60)

    # 生成数据
    torch.manual_seed(42)
    data = torch.tensor([150, 80, 200, 100, 170], dtype=torch.float32)

    print(f"\n原始数据：{data.tolist()}")

    # 标准化
    mean = data.mean()
    std = data.std()
    standardized = (data - mean) / std

    print(f"\n标准化（Standardization）：")
    print(f"  公式：x' = (x - μ) / σ")
    print(f"  均值 μ = {mean:.2f}，标准差 σ = {std:.2f}")
    print(f"  结果：{standardized.tolist()}")
    print(f"  新均值：{standardized.mean():.2f}，新标准差：{standardized.std():.2f}")

    # 归一化
    min_val = data.min()
    max_val = data.max()
    normalized = (data - min_val) / (max_val - min_val)

    print(f"\n归一化（Normalization）：")
    print(f"  公式：x' = (x - min) / (max - min)")
    print(f"  min = {min_val:.2f}，max = {max_val:.2f}")
    print(f"  结果：{normalized.tolist()}")
    print(f"  范围：[{normalized.min():.2f}, {normalized.max():.2f}]")


def main():
    print("=" * 60)
    print("数据标准化效果演示")
    print("=" * 60)

    # 1. 没有标准化
    demo_no_standardization()

    # 2. 标准化后
    demo_standardization()

    # 3. 对比
    demo_comparison()

    # 4. 标准化 vs 归一化
    demo_standardization_vs_normalization()

    # 总结
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("""
数据标准化的方法：

1. 标准化（Standardization）：
   x' = (x - μ) / σ
   效果：均值为0，标准差为1
   适用：大多数情况

2. 归一化（Normalization）：
   x' = (x - min) / (max - min)
   效果：范围 [0, 1]
   适用：数据有明确边界

为什么需要标准化？
  - 不同特征的尺度差异大
  - 尺度大的特征会主导梯度更新
  - 标准化后，所有特征同等重要
  - 训练更快、更稳定

注意事项：
  - 只在训练集上计算均值和标准差
  - 用训练集的统计量标准化测试集
  - 不能用测试集的统计量（否则是数据泄露）
""")


if __name__ == '__main__':
    main()
