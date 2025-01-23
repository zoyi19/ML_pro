import os
import cv2
import albumentations as A
from tqdm import tqdm

# 输入和输出路径
input_dir = "C:\\Users\\herr guo\\Desktop\\ML_project\\train_val\\train_val_data"
output_dir = "C:\\Users\\herr guo\\Desktop\\ML_project\\augmented_train_val_data"

# 确保输出路径存在
os.makedirs(output_dir, exist_ok=True)

# 定义增强管道
augmentation_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),  # 随机水平翻转
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),  # 随机亮度对比度调整
    A.Rotate(limit=15, p=0.5),  # 随机旋转
])

# 遍历图片文件
for class_dir in tqdm(os.listdir(input_dir), desc="Processing classes"):
    class_path = os.path.join(input_dir, class_dir)
    if not os.path.isdir(class_path):
        continue

    # 创建每个类别的输出子文件夹
    output_class_dir = os.path.join(output_dir, class_dir)
    os.makedirs(output_class_dir, exist_ok=True)

    for img_file in tqdm(os.listdir(class_path), desc=f"Processing images in {class_dir}", leave=False):
        if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        # 加载图片
        img_path = os.path.join(class_path, img_file)
        image = cv2.imread(img_path)

        # 检查图片是否成功加载
        if image is None:
            print(f"Warning: Unable to load image {img_file}")
            continue

        # 保存原图
        save_path_original = os.path.join(output_class_dir, f"original_{img_file}")
        cv2.imwrite(save_path_original, image)

        # 生成增强图片
        for i in range(2):  # 每张图片生成 2 个增强版本
            augmented = augmentation_pipeline(image=image)["image"]
            save_path_augmented = os.path.join(output_class_dir, f"aug_{i+1}_{img_file}")
            cv2.imwrite(save_path_augmented, augmented)

print("Data augmentation completed!")
