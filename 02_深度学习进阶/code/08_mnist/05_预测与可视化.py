"""
第2章 2.8 实战：手写数字识别
08_mnist/05_预测与可视化.py

功能：加载模型，预测并可视化结果
对应教程：08_实战_手写数字识别.md
"""
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class CNN(nn.Module):
    """CNN 模型"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def load_data():
    """加载测试数据"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=0)

    return test_loader


def main():
    print("=" * 50)
    print("MNIST 预测与可视化")
    print("=" * 50)

    # 设备
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"使用设备: {device}")

    # 加载模型
    model = CNN().to(device)
    try:
        model.load_state_dict(torch.load('best_model.pth', map_location=device))
        print("已加载模型")
    except FileNotFoundError:
        print("未找到模型文件，请先运行训练脚本")
        return

    model.eval()

    # 加载测试数据
    test_loader = load_data()
    test_images, test_labels = next(iter(test_loader))
    test_images = test_images.to(device)

    # 预测
    with torch.no_grad():
        output = model(test_images)
        predictions = output.argmax(dim=1)
        confidences = torch.softmax(output, dim=1).max(dim=1)[0]

    # 可视化预测结果
    fig, axes = plt.subplots(5, 5, figsize=(12, 12))
    fig.suptitle('MNIST 手写数字识别结果', fontsize=16)

    for i, ax in enumerate(axes.flat):
        if i < 25:
            ax.imshow(test_images[i].cpu().squeeze(), cmap='gray')

            pred = predictions[i].item()
            true = test_labels[i].item()
            conf = confidences[i].item()
            color = 'green' if pred == true else 'red'

            ax.set_title(f'预测: {pred} ({conf:.1%})\n真实: {true}',
                        color=color, fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('prediction_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("已保存预测结果到 prediction_results.png")

    # 统计
    correct = (predictions == test_labels[:25].to(device)).sum().item()
    print(f"\n前 25 张准确率: {correct}/25 ({100.*correct/25:.1f}%)")


if __name__ == '__main__':
    main()
