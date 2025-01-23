import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import json

# 配置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 数据路径
dataset_path = r"C:\Users\herr guo\Desktop\ML_project\augmented_train_val_data"

# 加载数据集
dataset = ImageFolder(root=dataset_path, transform=transform)
print(f"Classes: {dataset.classes}")
print(f"Number of samples: {len(dataset)}")

# 数据集划分
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

# -------------------------------
# 加载预训练的 DenseNet
# -------------------------------
# 加载 DenseNet-121 模型（预训练）
model = models.densenet121(pretrained=True)

# 替换分类头
num_classes = 12  # 新任务类别数
model.classifier = nn.Linear(model.classifier.in_features, num_classes)

# 迁移学习：冻结特定层（可选）
for param in model.features.parameters():
    param.requires_grad = False  # 冻结特征提取层

model = model.to(device)

# -------------------------------
# 定义损失函数和优化器
# -------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)  # 去掉 weight_decay 参数

# 自动混合精度
from torch.amp import GradScaler
scaler = GradScaler()

# -------------------------------
# 训练和验证
# -------------------------------
num_epochs = 20
train_loss_history = []
val_accuracy_history = []

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

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")
    train_loss_history.append(running_loss / len(train_loader))

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

# -------------------------------
# 保存模型权重和训练记录
# -------------------------------
torch.save(model.state_dict(), "densenet_finetune_weights_no_regularization.pth")
print("Model weights saved to densenet_finetune_weights_no_regularization.pth")

with open("training_metrics_no_regularization.json", "w") as f:
    json.dump({"train_loss": train_loss_history, "val_accuracy": val_accuracy_history}, f)
print("Training metrics saved to training_metrics_no_regularization.json")
