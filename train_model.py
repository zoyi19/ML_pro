import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import torch.nn as nn
import torch.optim as optim
from timm import create_model
import json

def train_model():
    # 配置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据路径
    dataset_path = r"C:\Users\herr guo\Desktop\ML_project\augmented_train_val_data"

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 加载数据集
    dataset = ImageFolder(root=dataset_path, transform=transform)
    print(f"Classes: {dataset.classes}")
    print(f"Number of samples: {len(dataset)}")

    # 数据集划分
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # 加载预训练模型并修改分类头
    model = create_model("resnet50", pretrained=True, num_classes=len(dataset.classes))
    model = model.to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # 训练和验证记录
    num_epochs = 20
    train_loss_history = []
    val_accuracy_history = []

    # 训练过程
    for epoch in range(num_epochs):
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

        avg_train_loss = running_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss:.4f}")

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
    torch.save(model.state_dict(), "model_weights.pth")
    print("Model weights saved to model_weights.pth")

    # 保存训练记录
    with open("training_metrics.json", "w") as f:
        json.dump({"train_loss": train_loss_history, "val_accuracy": val_accuracy_history}, f)
    print("Training metrics saved to training_metrics.json")

if __name__ == "__main__":
    train_model()
