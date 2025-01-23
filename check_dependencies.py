import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import torch.nn.functional as F

# 配置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图片大小
    transforms.ToTensor(),  # 转换为 PyTorch 张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
])

# 数据路径
dataset_path = r"C:\Users\herr guo\Desktop\ML_project\augmented_train_val_data"

# 加载数据集
dataset = ImageFolder(root=dataset_path, transform=transform)
print(f"Classes: {dataset.classes}")
print(f"Number of samples: {len(dataset)}")

# 数据集划分
train_size = int(0.8 * len(dataset))  # 80% 作为训练集
val_size = len(dataset) - train_size  # 20% 作为验证集
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

# 定义 Bottleneck 模块
class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(torch.relu(self.bn1(x)))
        out = self.conv2(torch.relu(self.bn2(out)))
        return torch.cat([x, out], 1)

# 定义 Dense Block
class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(Bottleneck(in_channels + i * growth_rate, growth_rate))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# 定义 Transition Layer
class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv(torch.relu(self.bn(x)))
        out = self.pool(out)
        return out

# 定义 DenseNet 模型
class DenseNet(nn.Module):
    def __init__(self, num_classes=12, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64):
        super(DenseNet, self).__init__()
        self.conv1 = nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_init_features)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.features = nn.Sequential()
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_layers, num_features, growth_rate)
            self.features.add_module(f'denseblock{i+1}', block)
            num_features += num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = Transition(num_features, num_features // 2)
                self.features.add_module(f'transition{i+1}', trans)
                num_features = num_features // 2

        self.bn_final = nn.BatchNorm2d(num_features)
        self.dropout = nn.Dropout(p=0.5)  # 添加 Dropout 层
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(torch.relu(self.bn1(x)))
        x = self.features(x)
        x = torch.relu(self.bn_final(x))
        x = torch.flatten(F.adaptive_avg_pool2d(x, (1, 1)), 1)  # 修正此部分
        x = self.dropout(x)  # 应用 Dropout
        x = self.fc(x)
        return x

# 导出模型为 ONNX 格式
def export_to_onnx(model, file_path="densenet_model.onnx"):
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    torch.onnx.export(
        model,                      # 模型
        dummy_input,                # 一个 dummy 输入
        file_path,                  # 保存路径
        export_params=True,         # 保存模型参数
        opset_version=11,           # ONNX opset 版本
        do_constant_folding=True,   # 是否执行常量折叠优化
        input_names=["input"],      # 输入名称
        output_names=["output"],    # 输出名称
        dynamic_axes={              # 可变维度（例如批量大小）
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }
    )
    print(f"ONNX model saved to {file_path}. You can open it with Netron.")

if __name__ == "__main__":
    # 初始化模型
    model = DenseNet(num_classes=12).to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # 自动混合精度
    from torch.amp import GradScaler
    scaler = GradScaler()

    # 训练模型
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # 验证模型
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Validation Accuracy: {100 * correct / total:.2f}%")

    # 导出 ONNX 模型
    export_to_onnx(model)
