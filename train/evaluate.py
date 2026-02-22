import os
import argparse
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from datapre.data_loader import get_data_loaders
from models import get_model

def plot_confusion_matrix(cm, classes, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    print(f"Confusion Matrix saved to {save_path}")
    plt.close()

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载数据 (只取 Test Loader)
    _, _, test_loader = get_data_loaders(args.data_root, batch_size=args.batch_size, num_workers=4)

    # 获取类别名称 (需要与 data_loader.py 中的一致)
    CLASSES = ['A', 'B', 'C', 'D', 'E']

    # 2. 加载模型
    print(f"Loading model: {args.model}")
    model = get_model(args.model, num_classes=5, in_channels=2)

    # 加载权重
    print(f"Loading weights from {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location=device)

    # 处理 DataParallel 带来的 'module.' 前缀问题
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    # 3. 推理
    all_preds = []
    all_labels = []

    print("Running Inference...")
    with torch.no_grad():
        for data, targets in tqdm(test_loader):
            data = data.to(device)
            outputs = model(data)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.numpy())

    # 4. 生成报告
    print("\n" + "="*30)
    print("Classification Report")
    print("="*30)
    print(classification_report(all_labels, all_preds, target_names=CLASSES, digits=4))

    # 5. 绘制混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    save_cm_path = os.path.join(os.path.dirname(args.checkpoint), f"confusion_matrix_{args.model}.png")
    plot_confusion_matrix(cm, CLASSES, save_cm_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate 3D CNN")

    parser.add_argument('--data_root', type=str, required=True, help='Path to dataset root')
    parser.add_argument('--model', type=str, default='resnet50', choices=['resnet50', 'resnet50_cbam', 'densenet121', 'vit'])
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to .pth model file')
    parser.add_argument('--batch_size', type=int, default=32)

    args = parser.parse_args()
    evaluate(args)