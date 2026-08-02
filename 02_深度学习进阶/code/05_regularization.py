"""
第2章 2.5 正则化
code/05_regularization.py

功能：演示 L1/L2 正则化、Dropout、Early Stopping 的效果
对应教程：05_正则化.md
"""
import torch
import torch.nn as nn
import torch.optim as optim


def create_model(dropout_rate=0.0):
    """创建带 Dropout 的模型"""

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(10, 64),
                nn.ReLU(),
                nn.Dropout(dropout_rate),   # Dropout
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(dropout_rate),   # Dropout
                nn.Linear(32, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.net(x)

    return Net()


def train_with_regularization(model, X_train, y_train, X_val, y_val,
                               epochs=500, lr=0.01, weight_decay=0.0,
                               use_l1=False, lambda_l1=0.01):
    """训练模型，支持 L1/L2 正则化"""

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCELoss()

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # 训练
        model.train()
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)

        # L1 正则化（手动添加）
        if use_l1:
            l1_loss = 0
            for param in model.parameters():
                l1_loss += torch.sum(torch.abs(param))
            loss = loss + lambda_l1 * l1_loss

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


def demo_overfitting():
    """演示过拟合问题"""
    print("=" * 60)
    print("1. 过拟合问题演示")
    print("=" * 60)

    # 生成数据
    torch.manual_seed(42)
    n_train = 20  # 少量训练样本
    n_val = 100
    n_features = 10

    X_train = torch.randn(n_train, n_features)
    y_train = (X_train.sum(dim=1) > 0).float().unsqueeze(1)

    X_val = torch.randn(n_val, n_features)
    y_val = (X_val.sum(dim=1) > 0).float().unsqueeze(1)

    # 训练
    model = create_model(dropout_rate=0.0)
    train_losses, val_losses = train_with_regularization(
        model, X_train, y_train, X_val, y_val,
        epochs=300, lr=0.01
    )

    # 输出结果
    print(f"\n训练样本：{n_train}，验证样本：{n_val}")
    print(f"\n训练过程：")
    print(f"  epoch 50:  train_loss={train_losses[49]:.4f}, val_loss={val_losses[49]:.4f}")
    print(f"  epoch 100: train_loss={train_losses[99]:.4f}, val_loss={val_losses[99]:.4f}")
    print(f"  epoch 200: train_loss={train_losses[199]:.4f}, val_loss={val_losses[199]:.4f}")
    print(f"  epoch 300: train_loss={train_losses[299]:.4f}, val_loss={val_losses[299]:.4f}")

    overfit = val_losses[-1] - train_losses[-1]
    print(f"\n过拟合程度：{overfit:+.4f}")
    print("→ 训练损失很低，但验证损失很高，说明模型'死记硬背'了训练数据")


def demo_l1_l2_comparison():
    """演示 L1 vs L2 正则化的效果"""
    print("\n" + "=" * 60)
    print("2. L1 vs L2 正则化对比")
    print("=" * 60)

    # 生成数据
    torch.manual_seed(42)
    n_train = 50
    n_val = 50
    n_features = 10

    X_train = torch.randn(n_train, n_features)
    y_train = (X_train.sum(dim=1) > 0).float().unsqueeze(1)

    X_val = torch.randn(n_val, n_features)
    y_val = (X_val.sum(dim=1) > 0).float().unsqueeze(1)

    print(f"\n数据：{n_train} 训练样本，{n_val} 验证样本，{n_features} 个特征")

    # 对比
    print("\n" + "-" * 70)
    print(f"{'方法':<20} {'训练损失':<12} {'验证损失':<12} {'过拟合程度':<15}")
    print("-" * 70)

    # 1. 无正则化
    torch.manual_seed(42)
    model = create_model(dropout_rate=0.0)
    train_l, val_l = train_with_regularization(
        model, X_train, y_train, X_val, y_val,
        epochs=500, lr=0.01, weight_decay=0.0
    )
    overfit = val_l[-1] - train_l[-1]
    print(f"{'无正则化':<20} {train_l[-1]:<12.4f} {val_l[-1]:<12.4f} {overfit:<+15.4f}")

    # 2. L2 正则化 (weight_decay)
    torch.manual_seed(42)
    model = create_model(dropout_rate=0.0)
    train_l, val_l = train_with_regularization(
        model, X_train, y_train, X_val, y_val,
        epochs=500, lr=0.01, weight_decay=0.01
    )
    overfit = val_l[-1] - train_l[-1]
    print(f"{'L2 正则化 (wd=0.01)':<20} {train_l[-1]:<12.4f} {val_l[-1]:<12.4f} {overfit:<+15.4f}")

    # 3. L1 正则化
    torch.manual_seed(42)
    model = create_model(dropout_rate=0.0)
    train_l, val_l = train_with_regularization(
        model, X_train, y_train, X_val, y_val,
        epochs=500, lr=0.01, weight_decay=0.0,
        use_l1=True, lambda_l1=0.01
    )
    overfit = val_l[-1] - train_l[-1]
    print(f"{'L1 正则化 (λ=0.01)':<20} {train_l[-1]:<12.4f} {val_l[-1]:<12.4f} {overfit:<+15.4f}")

    print("-" * 70)


def demo_sparsity():
    """演示 L1 的稀疏性效果"""
    print("\n" + "=" * 60)
    print("3. L1 正则化的稀疏性效果")
    print("=" * 60)

    # 生成数据
    torch.manual_seed(42)
    n_train = 50
    n_val = 50
    n_features = 10

    X_train = torch.randn(n_train, n_features)
    y_train = (X_train.sum(dim=1) > 0).float().unsqueeze(1)

    X_val = torch.randn(n_val, n_features)
    y_val = (X_val.sum(dim=1) > 0).float().unsqueeze(1)

    # 训练无正则化模型
    torch.manual_seed(42)
    model_no_reg = create_model(dropout_rate=0.0)
    train_with_regularization(
        model_no_reg, X_train, y_train, X_val, y_val,
        epochs=500, lr=0.01, weight_decay=0.0
    )

    # 训练 L1 正则化模型
    torch.manual_seed(42)
    model_l1 = create_model(dropout_rate=0.0)
    train_with_regularization(
        model_l1, X_train, y_train, X_val, y_val,
        epochs=500, lr=0.01, weight_decay=0.0,
        use_l1=True, lambda_l1=0.01
    )

    # 统计接近 0 的权重比例
    print("\n权重分布对比：")
    print("-" * 50)
    for name, model in [("无正则化", model_no_reg), ("L1 正则化", model_l1)]:
        all_weights = []
        for param in model.parameters():
            all_weights.extend(param.data.abs().view(-1).tolist())
        all_weights = torch.tensor(all_weights)
        near_zero = (all_weights.abs() < 0.01).float().mean().item()
        print(f"  {name}: 接近0的权重比例 = {near_zero:.1%}")

    print("\n→ L1 正则化让很多权重变为 0（稀疏性）")
    print("→ 这可以用于特征选择：权重为 0 的特征不重要")


def demo_dropout():
    """演示 Dropout 的效果"""
    print("\n" + "=" * 60)
    print("4. Dropout 效果演示")
    print("=" * 60)

    # 生成数据
    torch.manual_seed(42)
    n_train = 50
    n_val = 50
    n_features = 10

    X_train = torch.randn(n_train, n_features)
    y_train = (X_train.sum(dim=1) > 0).float().unsqueeze(1)

    X_val = torch.randn(n_val, n_features)
    y_val = (X_val.sum(dim=1) > 0).float().unsqueeze(1)

    print(f"\n数据：{n_train} 训练样本，{n_val} 验证样本，{n_features} 个特征")

    # 对比
    print("\n" + "-" * 70)
    print(f"{'方法':<20} {'训练损失':<12} {'验证损失':<12} {'过拟合程度':<15}")
    print("-" * 70)

    # 1. 无正则化
    torch.manual_seed(42)
    model = create_model(dropout_rate=0.0)
    train_l, val_l = train_with_regularization(
        model, X_train, y_train, X_val, y_val,
        epochs=500, lr=0.01
    )
    overfit = val_l[-1] - train_l[-1]
    print(f"{'无正则化':<20} {train_l[-1]:<12.4f} {val_l[-1]:<12.4f} {overfit:<+15.4f}")

    # 2. Dropout (p=0.3)
    torch.manual_seed(42)
    model = create_model(dropout_rate=0.3)
    train_l, val_l = train_with_regularization(
        model, X_train, y_train, X_val, y_val,
        epochs=500, lr=0.01
    )
    overfit = val_l[-1] - train_l[-1]
    print(f"{'Dropout (p=0.3)':<20} {train_l[-1]:<12.4f} {val_l[-1]:<12.4f} {overfit:<+15.4f}")

    # 3. Dropout (p=0.5)
    torch.manual_seed(42)
    model = create_model(dropout_rate=0.5)
    train_l, val_l = train_with_regularization(
        model, X_train, y_train, X_val, y_val,
        epochs=500, lr=0.01
    )
    overfit = val_l[-1] - train_l[-1]
    print(f"{'Dropout (p=0.5)':<20} {train_l[-1]:<12.4f} {val_l[-1]:<12.4f} {overfit:<+15.4f}")

    print("-" * 70)
    print("\n→ Dropout 通过随机丢弃神经元，减少过拟合")
    print("→ p=0.3 表示每个神经元有 30% 的概率被丢弃")


def demo_early_stopping():
    """演示 Early Stopping"""
    print("\n" + "=" * 60)
    print("5. Early Stopping 演示")
    print("=" * 60)

    # 生成数据
    torch.manual_seed(42)
    n_train = 50
    n_val = 50
    n_features = 10

    X_train = torch.randn(n_train, n_features)
    y_train = (X_train.sum(dim=1) > 0).float().unsqueeze(1)

    X_val = torch.randn(n_val, n_features)
    y_val = (X_val.sum(dim=1) > 0).float().unsqueeze(1)

    # Early Stopping 类
    class EarlyStopping:
        def __init__(self, patience=5, min_delta=0.001):
            self.patience = patience
            self.min_delta = min_delta
            self.counter = 0
            self.best_loss = None
            self.best_epoch = None

        def __call__(self, val_loss, epoch):
            if self.best_loss is None:
                self.best_loss = val_loss
                self.best_epoch = epoch
                return False
            elif val_loss < self.best_loss - self.min_delta:
                self.best_loss = val_loss
                self.best_epoch = epoch
                self.counter = 0
                return False
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    return True
                return False

    # 训练
    model = create_model(dropout_rate=0.0)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCELoss()

    early_stopping = EarlyStopping(patience=10)
    train_losses = []
    val_losses = []

    print("\n训练过程：")
    for epoch in range(200):
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

        # 检查是否停止
        if early_stopping(val_loss.item(), epoch):
            print(f"  epoch {epoch+1}: Early Stopping! (patience={early_stopping.patience})")
            print(f"  最佳 epoch: {early_stopping.best_epoch+1}，最佳 val_loss: {early_stopping.best_loss:.4f}")
            break

        # 每 50 个 epoch 输出一次
        if (epoch + 1) % 50 == 0:
            print(f"  epoch {epoch+1}: train_loss={train_losses[-1]:.4f}, val_loss={val_losses[-1]:.4f}")

    print("\n→ Early Stopping 在验证损失不再下降时停止训练")
    print(f"→ 保存了最佳模型（epoch {early_stopping.best_epoch+1}）")


def demo_combined():
    """演示多种正则化方法组合"""
    print("\n" + "=" * 60)
    print("6. 多种正则化方法组合")
    print("=" * 60)

    # 生成数据
    torch.manual_seed(42)
    n_train = 50
    n_val = 50
    n_features = 10

    X_train = torch.randn(n_train, n_features)
    y_train = (X_train.sum(dim=1) > 0).float().unsqueeze(1)

    X_val = torch.randn(n_val, n_features)
    y_val = (X_val.sum(dim=1) > 0).float().unsqueeze(1)

    print(f"\n数据：{n_train} 训练样本，{n_val} 验证样本，{n_features} 个特征")

    # 对比
    print("\n" + "-" * 70)
    print(f"{'方法':<25} {'训练损失':<12} {'验证损失':<12} {'过拟合程度':<15}")
    print("-" * 70)

    # 1. 无正则化
    torch.manual_seed(42)
    model = create_model(dropout_rate=0.0)
    train_l, val_l = train_with_regularization(
        model, X_train, y_train, X_val, y_val,
        epochs=500, lr=0.01
    )
    overfit = val_l[-1] - train_l[-1]
    print(f"{'无正则化':<25} {train_l[-1]:<12.4f} {val_l[-1]:<12.4f} {overfit:<+15.4f}")

    # 2. L2 + Dropout
    torch.manual_seed(42)
    model = create_model(dropout_rate=0.3)
    train_l, val_l = train_with_regularization(
        model, X_train, y_train, X_val, y_val,
        epochs=500, lr=0.01, weight_decay=0.01
    )
    overfit = val_l[-1] - train_l[-1]
    print(f"{'L2 + Dropout (p=0.3)':<25} {train_l[-1]:<12.4f} {val_l[-1]:<12.4f} {overfit:<+15.4f}")

    # 3. L2 + Dropout + L1
    torch.manual_seed(42)
    model = create_model(dropout_rate=0.3)
    train_l, val_l = train_with_regularization(
        model, X_train, y_train, X_val, y_val,
        epochs=500, lr=0.01, weight_decay=0.01,
        use_l1=True, lambda_l1=0.001
    )
    overfit = val_l[-1] - train_l[-1]
    print(f"{'L2 + Dropout + L1':<25} {train_l[-1]:<12.4f} {val_l[-1]:<12.4f} {overfit:<+15.4f}")

    print("-" * 70)
    print("\n→ 多种正则化方法可以组合使用，效果更好")


def main():
    print("=" * 60)
    print("正则化效果演示")
    print("=" * 60)

    # 1. 过拟合问题
    demo_overfitting()

    # 2. L1 vs L2 对比
    demo_l1_l2_comparison()

    # 3. 稀疏性
    demo_sparsity()

    # 4. Dropout
    demo_dropout()

    # 5. Early Stopping
    demo_early_stopping()

    # 6. 组合
    demo_combined()

    # 总结
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("""
正则化方法对比：

1. L2 正则化 (weight_decay):
   - 惩罚大权重，使权重变小
   - 实现简单：optimizer 参数 weight_decay
   - 适合大多数场景

2. L1 正则化:
   - 产生稀疏权重（很多变为 0）
   - 适合特征选择
   - 需要手动实现

3. Dropout:
   - 训练时随机丢弃神经元
   - 相当于集成多个子网络
   - 适合全连接层
   - 测试时必须关闭 (model.eval())

4. Early Stopping:
   - val_loss 不降就停
   - 简单有效

5. 数据增强:
   - 增加训练数据的多样性
   - 适合图像、文本等数据

如何选择？
   - 数据少：数据增强 + Early Stopping
   - 特征多：L1 正则化（特征选择）
   - 通用：L2 正则化 + Dropout + Early Stopping
""")


if __name__ == '__main__':
    main()
