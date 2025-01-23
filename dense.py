import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

#定义Bottleneck模块
class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return torch.cat([x, out], 1)  # 将输入和输出连接在一起

# 定义Transition Layer
class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = self.pool(out)
        return out

#定义Dense Block
class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(Bottleneck(in_channels + i * growth_rate, growth_rate))  # 每层输入通道数递增

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

#定义DenseNet-121
class DenseNet(nn.Module):
    def __init__(self, num_classes=12, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        # 初始卷积层
        self.conv1 = nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_init_features)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 构建 Dense Block 和 Transition Layer
        self.features = nn.Sequential()
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_layers, num_features, growth_rate)
            self.features.add_module(f'denseblock{i + 1}', block)
            num_features += num_layers * growth_rate
            if i != len(block_config) - 1:  # 最后一个 Dense Block 后没有 Transition Layer
                trans = Transition(num_features, num_features // 2)
                self.features.add_module(f'transition{i + 1}', trans)
                num_features = num_features // 2

        # 最后 BN + 全局平均池化
        self.bn_final = nn.BatchNorm2d(num_features)
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.features(x)
        x = F.relu(self.bn_final(x))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

#初始化DenseNet-121
# 创建模型实例
model = DenseNet(num_classes=12, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64)
# 查看模型结构
print(model)



#模型训练


# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 数据加载
dataset = ImageFolder(root="C:\\Users\\herr guo\\Desktop\\ML_project\\train_val\\train_val_data", transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

# 训练循环
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

for epoch in range(20):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        print(f"Epoch: {epoch}, Loss: {loss.item()}")



    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

# 验证模型
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Validation Accuracy: {100 * correct / total:.2f}%")


"""gpu版本
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

if __name__ == "__main__":
    # 初始化模型
    model = DenseNet(num_classes=12).to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)  # L2 正则化

    # 自动混合精度
    from torch.amp import GradScaler
    scaler = GradScaler()
    
    
    
    # 初始化记录列表
    train_loss_history = []  # 训练损失历史
    val_accuracy_history = []  # 验证准确率历史

    # 训练过程
    num_epochs = 20
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
            #print(f"Epoch: {epoch}, Loss: {loss.item()}")


        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
        train_loss_history.append(running_loss / len(train_loader))  # 保存训练损失
        
        # 验证过程
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_accuracy = 100 * correct / total
        val_accuracy_history.append(val_accuracy)  # 保存验证准确率
        print(f"Validation Accuracy: {100 * correct / total:.2f}%")

    # 保存模型权重
    torch.save(model.state_dict(), "densenet_model.pth")
    print("Model weights saved to densenet_model.pth")

    # 保存训练记录
    import json
    with open("training_metrics.json", "w") as f:
        json.dump({"train_loss": train_loss_history, "val_accuracy": val_accuracy_history}, f)
    print("Training metrics saved to training_metrics.json")


"""







#Test-code
"""
import torch
import pandas as pd
import os
from PIL import Image
from dense_gpu import DenseNet, transform, device  # 从训练程序中导入

# 加载模型
model = DenseNet(num_classes=12).to(device)
model.load_state_dict(torch.load("densenet_model.pth"))
model.eval()
print("Model loaded successfully.")

# 测试集图片目录
test_images_folder = "C:\\Users\\herr guo\\Desktop\\ML_project\\test_data\\test"

# 定义类别名称（需与训练时一致）
classes = ["Bicycle", "Bridge", "Bus", "Car", "Chimney", "Crosswalk", "Hydrant", 
           "Motorcycle", "Other", "Palm", "Stair", "Traffic Light"]

# 创建结果 DataFrame
results = []

# 遍历测试集目录中的所有 PNG 文件
image_files = sorted([f for f in os.listdir(test_images_folder) if f.endswith('.png')])

for image_name in image_files:
    image_path = os.path.join(test_images_folder, image_name)

    # 加载图片并预处理
    try:
        image = Image.open(image_path).convert("RGB")  # 确保是 RGB 模式
        image = transform(image).unsqueeze(0).to(device)  # 添加 batch 维度

        # 模型预测
        with torch.no_grad():
            logits = model(image)  # 模型输出 logits
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]  # Softmax 转换为概率

        # 构建一行数据：[图片名, 各类别预测概率]
        result_row = [image_name] + probabilities.tolist()
        results.append(result_row)

    except Exception as e:
        print(f"Error processing image {image_name}: {e}")

# 创建结果 DataFrame
columns = ["ImageName"] + classes
results_df = pd.DataFrame(results, columns=columns)

# 保存为 CSV 文件
output_csv_path = "test_results.csv"
results_df.to_csv(output_csv_path, index=False)
print(f"Results saved to {output_csv_path}")
"""


"""
test V2
import torch
import pandas as pd
import os
from PIL import Image
import numpy as np
from dense_gpu import DenseNet, val_augmentations, device  # 从训练程序中导入模型和设备

# 加载模型
model = DenseNet(num_classes=12).to(device)
model.load_state_dict(torch.load("improved_densenet_model.pth"))
model.eval()
print("Model loaded successfully.")

# 测试集图片目录
test_images_folder = r"C:\Users\herr guo\Desktop\ML_project\test_data\test"

# 定义类别名称（需与训练时一致）
classes = ["Bicycle", "Bridge", "Bus", "Car", "Chimney", "Crosswalk", "Hydrant", 
           "Motorcycle", "Other", "Palm", "Stair", "Traffic Light"]

# 创建结果列表
results = []

# 遍历测试集目录中的所有 PNG 文件
image_files = sorted([f for f in os.listdir(test_images_folder) if f.endswith('.png')])

for image_name in image_files:
    image_path = os.path.join(test_images_folder, image_name)

    # 加载图片并预处理
    try:
        image = Image.open(image_path).convert("RGB")  # 确保是 RGB 模式
        image = val_augmentations(image=np.array(image))["image"].float()  # 确保数据类型为 float
        image = image.unsqueeze(0).to(device)  # 添加 batch 维度

        # 模型预测
        with torch.no_grad():
            logits = model(image)  # 模型输出 logits
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]  # Softmax 转换为概率

        # 构建一行数据：[图片名, 各类别预测概率]
        result_row = [image_name] + probabilities.tolist()
        results.append(result_row)

    except Exception as e:
        print(f"Error processing image {image_name}: {e}")

# 创建结果 DataFrame
columns = ["ImageName"] + classes
results_df = pd.DataFrame(results, columns=columns)

# 保存为 CSV 文件
output_csv_path = r"C:\Users\herr guo\Desktop\ML_project\test_results.csv"
results_df.to_csv(output_csv_path, index=False)
print(f"Results saved to {output_csv_path}")
"""


"""
train V2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F
import numpy as np

# 配置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 数据增强
train_augmentations = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    ToTensorV2()
])

val_augmentations = A.Compose([
    A.Resize(224, 224),
    ToTensorV2()
])

# 自定义数据集类，支持 Albumentations
from torchvision.datasets.folder import default_loader

class AlbumentationsDataset(ImageFolder):
    def __init__(self, root, transform=None, loader=default_loader):
        super().__init__(root, loader=loader)
        self.transform = transform

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = self.loader(path)
        if self.transform:
            # 添加 NumPy 转换，并确保数据类型为 float32
            image = self.transform(image=np.array(image))['image'].float()
        return image, label
# 数据路径
dataset_path = r"C:\Users\herr guo\Desktop\ML_project\augmented_train_val_data"

# 加载数据集
dataset = AlbumentationsDataset(root=dataset_path, transform=train_augmentations)
print(f"Classes: {dataset.classes}")
print(f"Number of samples: {len(dataset)}")

# 数据集划分
train_size = int(0.8 * len(dataset))  # 80% 作为训练集
val_size = len(dataset) - train_size  # 20% 作为验证集
train_dataset, val_dataset = random_split(
    dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

# 验证集使用不同的增强
val_dataset.dataset.transform = val_augmentations

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

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
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(torch.relu(self.bn1(x)))
        x = self.features(x)
        x = torch.relu(self.bn_final(x))
        x = torch.flatten(F.adaptive_avg_pool2d(x, (1, 1)), 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    # 初始化模型
    model = DenseNet(num_classes=12).to(device)

    # 类别权重
    class_weights = torch.tensor([1.0, 1.2, 0.8, 1.0, 1.5, 1.0, 2.0, 2.0, 1.0, 1.0, 1.2, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 优化器和调度器
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # 自动混合精度
    from torch.amp import GradScaler
    scaler = GradScaler()

    train_loss_history = []
    val_accuracy_history = []

    # 训练过程
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
        train_loss_history.append(running_loss / len(train_loader))
        scheduler.step(running_loss / len(train_loader))

        # 验证过程
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        val_accuracy_history.append(val_accuracy)
        print(f"Validation Accuracy: {val_accuracy:.2f}%")

    # 保存模型权重
    torch.save(model.state_dict(), "improved_densenet_model.pth")
    print("Model weights saved to improved_densenet_model.pth")

    # 保存训练记录
    import json
    with open("training_metrics.json", "w") as f:
        json.dump({"train_loss": train_loss_history, "val_accuracy": val_accuracy_history}, f)
    print("Training metrics saved to training_metrics.json")
"""