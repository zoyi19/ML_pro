from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# 设置路径
data_dir = r"C:\Users\herr guo\Desktop\ML_project\train_val\train_val_data"

# 加载数据
dataset = ImageFolder(data_dir)

# 检查类别
# print("Classes:", dataset.classes)  # 输出类别列表
# print("Sample:", dataset[0])       # 输出第一个样本

# import albumentations as A
# from albumentations.pytorch import ToTensorV2

# # 定义增强管道
# transform = A.Compose([
#     A.Resize(224, 224),                  # 将所有图像调整为 224x224
#     A.HorizontalFlip(p=0.5),             # 随机水平翻转，概率 50%
#     A.RandomBrightnessContrast(p=0.2),   # 随机调整亮度和对比度
#     A.HueSaturationValue(p=0.2),         # 随机调整色调、饱和度和亮度
#     A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # 标准化
#     ToTensorV2()                         # 转换为 PyTorch 张量
# ])

# import cv2
# import os
# from torch.utils.data import Dataset

# class CustomDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.dataset = ImageFolder(root_dir)  # 使用 ImageFolder 加载数据
#         self.transform = transform

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         img_path, label = self.dataset.imgs[idx]  # 获取图像路径和标签
#         image = cv2.imread(img_path)             # 加载图像
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式
        
#         if self.transform:
#             image = self.transform(image=image)["image"]
        
#         return image, label
# from torch.utils.data import DataLoader

# # 自定义数据集
# train_dataset = CustomDataset(data_dir, transform=transform)

# # 数据加载器
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# # 测试加载器
# for images, labels in train_loader:
#     print(images.shape, labels.shape)  # (batch_size, 3, 224, 224), (batch_size,)
#     break

# from torch.utils.data import random_split

# # 划分数据集：80% 训练，20% 验证
# train_size = int(0.8 * len(train_dataset))
# val_size = len(train_dataset) - train_size
# train_set, val_set = random_split(train_dataset, [train_size, val_size])

# # 加载训练和验证数据
# train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

# import matplotlib.pyplot as plt

# # 从数据集中取一批样本
# images, labels = next(iter(train_loader))

# # 可视化增强结果
# fig, axes = plt.subplots(1, 4, figsize=(16, 4))
# for i in range(4):
#     img = images[i].permute(1, 2, 0).numpy()  # 变换为 (H, W, C)
#     img = (img * 0.229 + 0.485).clip(0, 1)    # 反归一化
#     axes[i].imshow(img)
#     axes[i].set_title(f"Label: {labels[i]}")
#     axes[i].axis("off")
# plt.show()


import albumentations as A
import cv2
import os
from tqdm import tqdm

# 定义增强管道
transform = A.Compose([
    A.Resize(224, 224),                  # 调整大小
    A.HorizontalFlip(p=0.5),             # 随机水平翻转
    A.RandomBrightnessContrast(p=0.2),   # 调整亮度和对比度
    A.HueSaturationValue(p=0.2),         # 调整色调、饱和度和亮度
])

# 原始图片路径
input_folder = r"C:\Users\herr guo\Desktop\ML_project\train_val\train_val_data"

# 遍历每个类别文件夹
for class_name in tqdm(os.listdir(input_folder)):
    class_path = os.path.join(input_folder, class_name)
    if not os.path.isdir(class_path):  # 跳过非文件夹
        continue

    # 遍历类别文件夹中的每张图片
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):  # 跳过非图片文件
            continue

        # 读取图像
        image = cv2.imread(img_path)
        if image is None:  # 跳过无法读取的图片
            print(f"Warning: Cannot read image {img_path}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式

        # 增强多次以生成多个变体
        for i in range(5):  # 每张图片生成 5 个增强样本
            augmented = transform(image=image)
            augmented_image = augmented["image"]

            # 保存增强后的图片到原始文件夹
            augmented_img_name = f"{os.path.splitext(img_name)[0]}_aug_{i}.png"
            augmented_img_path = os.path.join(class_path, augmented_img_name)
            cv2.imwrite(augmented_img_path, cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))
