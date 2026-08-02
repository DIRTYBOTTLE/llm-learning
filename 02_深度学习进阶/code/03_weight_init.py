"""
第2章 2.3 权重初始化
code/03_weight_init.py

功能：演示不同初始化方法对训练的影响
对应教程：03_权重初始化.md
"""
import torch
import torch.nn as nn


def create_model(init_method):
    """创建一个简单网络并应用指定的初始化"""

    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(10, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.layers(x)

    model = SimpleNet()

    # 应用初始化（原地修改权重）
    for layer in model.layers:
        if isinstance(layer, nn.Linear):
            init_method(layer.weight)
            # 偏置也初始化
            nn.init.zeros_(layer.bias)

    return model


def train_model(model, X, y, epochs=200, lr=0.1):
    """训练模型并返回损失历史"""
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    losses = []
    for epoch in range(epochs):
        y_pred = model(X)
        loss = loss_fn(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return losses


def main():
    print("=" * 60)
    print("权重初始化对比实验")
    print("=" * 60)

    # ===== 生成数据 =====
    torch.manual_seed(42)
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100, 1)).float()

    print(f"\n数据：{X.shape[0]} 个样本，{X.shape[1]} 个特征")
    print(f"标签分布：{int(y.sum())} 个正样本，{int(len(y)-y.sum())} 个负样本")

    # ===== 定义不同的初始化方法 =====
    # 注意：必须用原地操作，不能用 w.clone()
    def init_zeros(w):
        nn.init.zeros_(w)

    def init_small(w):
        nn.init.normal_(w, mean=0, std=0.01)

    def init_large(w):
        nn.init.normal_(w, mean=0, std=1.0)

    def init_xavier(w):
        nn.init.xavier_uniform_(w)

    def init_he(w):
        nn.init.kaiming_uniform_(w, nonlinearity='relu')

    init_methods = {
        "全零初始化": init_zeros,
        "太小 (σ=0.01)": init_small,
        "太大 (σ=1.0)": init_large,
        "Xavier": init_xavier,
        "He (推荐)": init_he,
    }

    # ===== 对比训练 =====
    print("\n" + "-" * 60)
    print(f"{'初始化方法':<15} {'最终损失':>10} {'最终准确率':>10} {'是否收敛':>10}")
    print("-" * 60)

    results = {}
    for name, init_fn in init_methods.items():
        torch.manual_seed(42)
        model = create_model(init_fn)
        losses = train_model(model, X, y, epochs=200, lr=0.1)

        with torch.no_grad():
            y_pred = model(X)
            acc = ((y_pred > 0.5).float() == y).float().mean().item()

        final_loss = losses[-1]
        converged = final_loss < 0.5

        results[name] = {
            'losses': losses,
            'final_loss': final_loss,
            'accuracy': acc,
            'converged': converged
        }

        status = "✓ 收敛" if converged else "✗ 未收敛"
        print(f"{name:<15} {final_loss:>10.4f} {acc:>9.1%} {status:>10}")

    print("-" * 60)

    # ===== 查看初始化后的权重分布 =====
    print("\n" + "=" * 60)
    print("初始化后的权重分布（第一层）")
    print("=" * 60)

    torch.manual_seed(42)
    for name, init_fn in init_methods.items():
        model = create_model(init_fn)
        first_layer = model.layers[0]
        w = first_layer.weight.data
        print(f"\n{name}:")
        print(f"  均值: {w.mean():.4f}, 标准差: {w.std():.4f}")
        print(f"  范围: [{w.min():.4f}, {w.max():.4f}]")


if __name__ == '__main__':
    main()
