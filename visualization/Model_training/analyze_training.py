import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ================= 配置区域 =================
ROOT_DIR = '../checkpoints_cbam'  # 数据根目录
OUTPUT_DIR = ''  # 结果保存目录

# 绘图设置
FIG_DPI = 300  # 高分辨率
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial'] # 适配中文和英文
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
# ===========================================

def load_all_data(root_dir):
    """
    遍历目录，加载所有模型的训练日志
    """
    data_map = {}
    if not os.path.exists(root_dir):
        print(f"错误: 目录 {root_dir} 不存在")
        return data_map

    subfolders = sorted(os.listdir(root_dir))
    for folder in subfolders:
        folder_path = os.path.join(root_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        # 获取模型名称 (文件夹名)
        model_name = folder

        df = None
        # 1. 优先读取 metrics.csv
        metrics_path = os.path.join(folder_path, 'metrics.csv')
        if os.path.exists(metrics_path):
            try:
                df = pd.read_csv(metrics_path)
            except Exception:
                pass

        # 2. 如果没有，尝试读取 train.log (csv格式)
        if df is None:
            logs = glob.glob(os.path.join(folder_path, '*train.log'))
            for log in logs:
                try:
                    temp = pd.read_csv(log)
                    # 简单检查是否包含必要的列
                    cols = [str(c).lower() for c in temp.columns]
                    if any('epoch' in c for c in cols):
                        df = temp
                        break
                except Exception:
                    continue

        if df is not None:
            # 标准化列名
            df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

            # 常用列名映射
            rename_map = {
                'acc': 'train_acc',
                'loss': 'train_loss',
                'train_accuracy': 'train_acc',
                'val_accuracy': 'val_acc',
                'accuracy': 'train_acc'
            }
            df = df.rename(columns=rename_map)

            # 确保有 epoch 列
            if 'epoch' not in df.columns:
                df['epoch'] = range(1, len(df) + 1)

            data_map[model_name] = df
            print(f"已加载: {model_name}")

    return data_map

def get_best_metrics(df):
    """
    获取单个模型 "最后一次保存" (通常是 Best) 时的指标
    优先级: Max Val F1 > Max Val Acc > Min Val Loss
    """
    best_row = None

    # 策略: 寻找最佳指标对应的行
    if 'val_f1' in df.columns:
        idx = df['val_f1'].idxmax()
        best_row = df.iloc[idx]
    elif 'val_acc' in df.columns:
        idx = df['val_acc'].idxmax()
        best_row = df.iloc[idx]
    elif 'val_loss' in df.columns:
        idx = df['val_loss'].idxmin()
        best_row = df.iloc[idx]
    else:
        # 如果没有验证集指标，取最后一行
        best_row = df.iloc[-1]

    return best_row

def plot_training_curves(data_map):
    """
    图1: 画每个模型的详细训练过程 (Loss/Acc/F1 曲线)
    生成一张超高分辨率的大图
    """
    models = sorted(list(data_map.keys()))
    n_models = len(models)
    if n_models == 0: return

    # 动态调整高度
    fig, axes = plt.subplots(n_models, 3, figsize=(20, 4 * n_models), dpi=FIG_DPI)
    if n_models == 1: axes = axes.reshape(1, -1)

    for i, name in enumerate(models):
        df = data_map[name]

        # Loss
        ax = axes[i][0]
        if 'train_loss' in df.columns: ax.plot(df['epoch'], df['train_loss'], label='Train')
        if 'val_loss' in df.columns: ax.plot(df['epoch'], df['val_loss'], label='Val', linestyle='--')
        ax.set_title(f"{name}\nLoss Curve", fontsize=12, fontweight='bold')
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Acc
        ax = axes[i][1]
        if 'train_acc' in df.columns: ax.plot(df['epoch'], df['train_acc'], label='Train')
        if 'val_acc' in df.columns: ax.plot(df['epoch'], df['val_acc'], label='Val', linestyle='--')
        ax.set_title(f"{name}\nAccuracy Curve", fontsize=12, fontweight='bold')
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # F1 / LR
        ax = axes[i][2]
        if 'val_f1' in df.columns:
            ax.plot(df['epoch'], df['val_f1'], color='green', label='Val F1')
            ax.set_ylabel("F1 Score")
            ax.set_title(f"{name}\nF1 Score Curve", fontsize=12, fontweight='bold')
        elif 'lr' in df.columns:
            ax.plot(df['epoch'], df['lr'], color='purple', label='LR')
            ax.set_ylabel("Learning Rate")
            ax.set_title(f"{name}\nLearning Rate", fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, '1_training_curves_detailed.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"图1已保存: {save_path}")

def plot_best_performance_comparison(data_map):
    """
    图2: 对比各个模型“最后一次保存（最佳状态）”的训练效果
    使用柱状图 (Bar Chart) 进行直观对比
    """
    models = sorted(list(data_map.keys()))
    if not models: return

    # 提取每个模型的最佳指标
    summary_data = []
    for m in models:
        row = get_best_metrics(data_map[m])
        # 提取关键指标，如果不存在则设为0
        item = {
            'Model': m,
            'Val Acc': row.get('val_acc', 0),
            'Val F1': row.get('val_f1', 0),
            'Val Loss': row.get('val_loss', 0),
            'Train Loss': row.get('train_loss', 0)
        }
        summary_data.append(item)

    df_sum = pd.DataFrame(summary_data)

    # 简化模型名称显示（去掉日期后缀，防止X轴太拥挤）
    # 假设格式如 model_name_2025...
    df_sum['ShortName'] = df_sum['Model'].apply(lambda x: x.split('_202')[0] if '_202' in x else x)

    # 定义要对比的指标
    metrics_to_plot = [
        ('Val Acc', 'Best Validation Accuracy (Higher is Better)'),
        ('Val F1', 'Best Validation F1 Score (Higher is Better)'),
        ('Val Loss', 'Best Validation Loss (Lower is Better)'),
        ('Train Loss', 'Final Training Loss (Lower is Better)')
    ]

    # 创建 2x2 的子图布局
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), dpi=FIG_DPI)
    axes = axes.flatten()

    # 颜色调色板
    colors = sns.color_palette("viridis", len(df_sum))

    for i, (metric, title) in enumerate(metrics_to_plot):
        ax = axes[i]
        # 画柱状图
        bars = ax.bar(df_sum['ShortName'], df_sum[metric], color=colors, alpha=0.8, edgecolor='black')

        # 设置标题和标签
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        # 在柱子上标注数值
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

        # 如果是 Loss，Y轴可能需要反转或者不做处理（Loss越低越好，柱子越短越好）
        # 这里保持默认，柱子短代表 Loss 低

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, '2_best_model_performance_comparison.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    # 同时也保存一份 csv 表格
    csv_path = os.path.join(OUTPUT_DIR, 'best_model_performance.csv')
    df_sum.to_csv(csv_path, index=False)
    print(f"图2已保存: {save_path}")
    print(f"性能对比表格已保存: {csv_path}")

if __name__ == "__main__":
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("正在分析...")
    all_data = load_all_data(ROOT_DIR)

    if all_data:
        # 画图1: 详细曲线
        plot_training_curves(all_data)

        # 画图2: 最佳效果对比 (柱状图)
        plot_best_performance_comparison(all_data)

        print("\n分析完成！请查看 analysis_results_v2 文件夹。")
    else:
        print("未找到有效数据。")