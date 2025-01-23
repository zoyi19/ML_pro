import torch
import pandas as pd
import os
from torchvision.transforms import transforms
from PIL import Image
from timm import create_model
from datetime import datetime

def test_model():
    # 配置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 测试集路径
    test_images_folder = r"C:\Users\herr guo\Desktop\ML_project\test_data\test"

    # 定义类别名称
    classes = ["Bicycle", "Bridge", "Bus", "Car", "Chimney", "Crosswalk", "Hydrant",
               "Motorcycle", "Other", "Palm", "Stair", "Traffic Light"]

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 加载模型
    model = create_model("resnet50", pretrained=False, num_classes=len(classes))
    model.load_state_dict(torch.load("model_weights.pth"))
    model = model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # 遍历测试集并生成预测结果
    results = []
    image_files = sorted([f for f in os.listdir(test_images_folder) if f.lower().endswith('.png')])

    for image_name in image_files:
        image_path = os.path.join(test_images_folder, image_name)

        try:
            # 加载图片并预处理
            image = Image.open(image_path).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)

            # 模型预测
            with torch.no_grad():
                logits = model(image)
                probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]

            # 保存预测结果
            result_row = [image_name] + probabilities.tolist()
            results.append(result_row)

        except Exception as e:
            print(f"Error processing image {image_name}: {e}")

    # 保存结果到 CSV 文件
    columns = ["ImageName"] + classes
    results_df = pd.DataFrame(results, columns=columns)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv_path = f"test_results_{timestamp}.csv"
    results_df.to_csv(output_csv_path, index=False)
    print(f"Results saved to {output_csv_path}")

if __name__ == "__main__":
    test_model()
