import os
import argparse
import datetime
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import pandas as pd # 推荐使用 pandas 记录 CSV

# 导入之前定义的模块
from datapre.data_loader import get_data_loaders
from models import get_model

# --- 新增日志工具类 ---
class TrainLogger:
    def __init__(self, base_dir, model_name):
        # 创建以时间命名的实验文件夹
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = os.path.join(base_dir, f"{model_name}_{timestamp}")
        os.makedirs(self.exp_dir, exist_ok=True)

        # 1. 设置文本日志 (logging)
        self.log_file = os.path.join(self.exp_dir, "train.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler() # 同时输出到控制台
            ]
        )
        self.logger = logging.getLogger()

        # 2. 设置 CSV 指标记录
        self.csv_file = os.path.join(self.exp_dir, "metrics.csv")
        self.metrics_history = []

    def log_metrics(self, epoch, train_loss, train_acc, val_loss, val_acc, val_f1, lr):
        # 存入历史列表
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "lr": lr
        }
        self.metrics_history.append(metrics)

        # 保存为 CSV
        df = pd.DataFrame(self.metrics_history)
        df.to_csv(self.csv_file, index=False)

        # 写入文本日志
        self.logger.info(
            f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f} | LR: {lr:.6f}"
        )

    def info(self, message):
        self.logger.info(message)

# --- 训练核心逻辑保持不变 ---
def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    loop = tqdm(loader, desc="Training", leave=False)
    for data, targets in loop:
        data = data.to(device)
        targets = targets.to(device)

        with autocast():
            outputs = model(data)
            loss = criterion(outputs, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * data.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(targets.cpu().numpy())

        loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, targets in tqdm(loader, desc="Validating", leave=False):
            data = data.to(device)
            targets = targets.to(device)

            with autocast():
                outputs = model(data)
                loss = criterion(outputs, targets)

            running_loss += loss.item() * data.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')

    return epoch_loss, epoch_acc, epoch_f1

def main(args):
    # 初始化日志器
    logger = TrainLogger(args.save_dir, args.model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 记录训练参数
    logger.info(f"Arguments: {vars(args)}")

    # 2. 数据加载
    logger.info("Loading data...")
    train_loader, val_loader, _ = get_data_loaders(
        args.data_root, batch_size=args.batch_size, num_workers=8
    )

    # 3. 模型初始化
    model = get_model(args.model, num_classes=5, in_channels=2)
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1 = 0.0

    # 5. 训练循环
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # 使用 Logger 统一保存指标并打印
        logger.log_metrics(epoch+1, train_loss, train_acc, val_loss, val_acc, val_f1, current_lr)

        # 保存最佳模型到当前实验文件夹
        if val_f1 > best_f1:
            best_f1 = val_f1
            save_path = os.path.join(logger.exp_dir, f"{args.model}_best.pth")
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_dict, save_path)
            logger.info(f">>> Saved New Best Model (F1: {best_f1:.4f})")

    logger.info("Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 3D CNN with Auto-Logging")
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--model', type=str, default='resnet50', choices=['resnet50', 'resnet18', 'resnet18_cbam', 'resnet50_cbam', 'densenet121','densenet121_cbam','vit'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=36)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_dir', type=str, default='./experiments', help='Root directory for all experiments')

    args = parser.parse_args()
    main(args)