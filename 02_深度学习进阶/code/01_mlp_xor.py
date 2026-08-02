"""
第2章 2.1 多层感知机
code/01_mlp_xor.py

功能：用 MLP 解决 XOR 问题，展示多层网络的能力
对应教程：01_多层感知机.md
"""
import torch
import torch.nn as nn


class MLP_XOR(nn.Module):
    """解决 XOR 问题的 MLP"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def train_once(X, y, seed):
    """训练一次，返回最终 loss"""
    torch.manual_seed(seed)
    model = MLP_XOR()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    loss_fn = nn.BCELoss()

    for epoch in range(2000):
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 计算最终准确率
    with torch.no_grad():
        y_pred = model(X)
        acc = ((y_pred > 0.5).float() == y).float().mean().item()
    return acc, model


def main():
    print("=" * 50)
    print("MLP 解决 XOR 问题")
    print("=" * 50)

    # XOR 数据
    X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

    print("\nXOR 问题：")
    for i in range(4):
        print(f"  输入: {X[i].tolist()} → 输出: {y[i].item():.0f}")

    # 多次训练，取最佳结果
    print("\n训练中（尝试10次不同初始化）...")
    best_acc = 0
    best_model = None

    for seed in range(10):
        acc, model = train_once(X, y, seed)
        if acc > best_acc:
            best_acc = acc
            best_model = model
        if acc == 1.0:
            print(f"  第 {seed+1} 次尝试：100% 准确！")
            break

    # 测试最佳模型
    print(f"\n最佳准确率: {best_acc:.0%}")

    print("\n预测结果：")
    print("-" * 40)
    with torch.no_grad():
        for i in range(4):
            pred = best_model(X[i:i+1])
            pred_label = int(pred.item() > 0.5)
            true_label = int(y[i].item())
            status = "✓" if pred_label == true_label else "✗"
            print(f"  输入: {X[i].tolist()} → 预测: {pred.item():.3f} → "
                  f"标签: {pred_label} (真实: {true_label}) {status}")

    print("-" * 40)

    # 隐藏层可视化
    print("\n隐藏层的作用：")
    print("原始数据线性不可分，隐藏层将其变换为线性可分：")

    with torch.no_grad():
        hidden = best_model.net[0](X)
        hidden = best_model.net[1](hidden)

        print("\n  原始输入 → 隐藏层输出（ReLU后）：")
        for i in range(4):
            print(f"    {X[i].tolist()} → {[round(v,3) for v in hidden[i].tolist()]}")


if __name__ == '__main__':
    main()
