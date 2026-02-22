import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.feature_selection import VarianceThreshold
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 尝试导入 plotly
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# ==========================================
# 0. 配置参数
# ==========================================
BATCH_SIZE = 32
NUM_WORKERS = 8
PIN_MEMORY = True
PREFETCH_FACTOR = 4

# 路径
DATA_ROOT = "/home/xxge/tzy/Pycharmpro/cnn_fiber/TRAIN_DATA_FINAL_256"
CHECKPOINT_ROOT = "./checkpoints_cbam"
OUTPUT_DIR = "tsne_3d_final_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 配色
CUSTOM_PALETTE = ['#EB6969', '#5D8AA8', '#89AA7B', '#B07AA1', '#E3AE57']
LABEL_MAPPING = {'E': 'CEL', 'L': 'LIP', 'LQ': 'LL', 'Q': 'LAC', 'QLX': 'LLC'}

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5
sys.path.append(os.getcwd())

# ==========================================
# 1. 健壮的数据加载器
# ==========================================
class FiberDataset(Dataset):
    def __init__(self, root_dir, target_depth=64):
        self.file_list = []
        self.target_depth = target_depth
        if not os.path.exists(root_dir): return
        subdirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        for folder_name in subdirs:
            folder_path = os.path.join(root_dir, folder_name)
            display_label = LABEL_MAPPING.get(folder_name, folder_name)
            files = [f for f in os.listdir(folder_path) if f.endswith('.npz')]
            for fname in files:
                self.file_list.append((os.path.join(folder_path, fname), display_label))

    def __len__(self): return len(self.file_list)

    def __getitem__(self, idx):
        fpath, label = self.file_list[idx]
        try:
            with np.load(fpath) as f:
                raw = f['data'] if 'data' in f else f[list(f.keys())[0]]
            if raw.ndim == 5: raw = raw[0]

            vol = raw.astype(np.float32)
            # 安全归一化
            mx = vol.max()
            if mx > 1e-6: vol /= mx
            else: vol[:] = 0
            vol = np.nan_to_num(vol)

            d = vol.shape[0]
            if d > self.target_depth:
                start = (d - self.target_depth) // 2
                vol = vol[start:start+self.target_depth]
            elif d < self.target_depth:
                pad = self.target_depth - d
                vol = np.pad(vol, ((pad//2, pad - pad//2), (0,0), (0,0), (0,0)), mode='constant')
            vol = vol.transpose(3, 0, 1, 2)
            return torch.from_numpy(vol), label
        except:
            return torch.zeros((1, self.target_depth, 256, 256)), "Error"

# ==========================================
# 2. 智能模型加载 (修复版)
# ==========================================
def find_checkpoints(root_dir):
    ckpts = []
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if f.endswith('.pth'): ckpts.append(os.path.join(root, f))
    return sorted(ckpts)

def load_model_smart(ckpt_path, num_classes=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 获取文件名全小写，用于判断
    fname = os.path.basename(ckpt_path).lower()
    folder = os.path.basename(os.path.dirname(ckpt_path)).lower()
    full_name = f"{folder}_{fname}"

    # 1. 判断是否使用 CBAM
    use_cbam = False
    if 'cbam' in full_name:
        use_cbam = True
        print(f"   🔥 检测到 CBAM 模块")

    model = None
    try:
        # 2. 精确区分架构
        if 'resnet18' in full_name:
            print(f"   ⚙️ 架构: ResNet18 (use_cbam={use_cbam})")
            from models.resnet3d import resnet18_3d
            model = resnet18_3d(num_classes=num_classes, use_cbam=use_cbam)

        elif 'resnet50' in full_name:
            print(f"   ⚙️ 架构: ResNet50 (use_cbam={use_cbam})")
            from models.resnet3d import resnet50_3d
            model = resnet50_3d(num_classes=num_classes, use_cbam=use_cbam)

        elif 'densenet' in full_name:
            print(f"   ⚙️ 架构: DenseNet121 (use_cbam={use_cbam})")
            from models.densenet3d import densenet121_3d
            model = densenet121_3d(num_classes=num_classes, use_cbam=use_cbam)

        elif 'vit' in full_name:
            print(f"   ⚙️ 架构: ViT3D")
            from models.vit3d import ViT3D
            model = ViT3D(num_classes=num_classes)

        else:
            # 默认情况，尝试用 ResNet50
            print(f"   ❓ 未知架构，尝试 ResNet50")
            from models.resnet3d import resnet50_3d
            model = resnet50_3d(num_classes=num_classes, use_cbam=use_cbam)

        # 3. 加载权重
        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

        # 处理 module. 前缀
        new_state = {k.replace('module.', ''): v for k, v in state_dict.items()}

        # 关键：加载权重，允许非严格匹配（防止一些辅助层报错）
        msg = model.load_state_dict(new_state, strict=False)
        # 打印一下丢失的键，如果关键层丢失会有提示
        if len(msg.missing_keys) > 0:
            # 过滤掉 FC 层的丢失警告，因为我们本来就要换掉它
            real_missing = [k for k in msg.missing_keys if 'fc' not in k and 'classifier' not in k and 'head' not in k]
            if real_missing:
                print(f"   ⚠️ 警告: 丢失部分权重: {real_missing[:3]}...")

        # 4. 移除分类头 (Feature Extraction)
        if hasattr(model, 'fc'): model.fc = nn.Identity()
        elif hasattr(model, 'classifier'): model.classifier = nn.Identity()
        elif hasattr(model, 'head'): model.head = nn.Identity()

        model.to(device)
        model.eval()
        return model, device

    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        return None, None

# ==========================================
# 3. 主程序
# ==========================================
def main():
    print(f"🚀 启动 3D t-SNE (模型修复版)...")
    dataset = FiberDataset(DATA_ROOT)
    if len(dataset) == 0: return

    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY, persistent_workers=True, prefetch_factor=PREFETCH_FACTOR
    )
    print(f"✅ 数据: {len(dataset)} 样本")

    checkpoints = find_checkpoints(CHECKPOINT_ROOT)
    unique_labels = sorted(list(set([item[1] for item in dataset.file_list])))
    label_to_color = {lbl: CUSTOM_PALETTE[i % len(CUSTOM_PALETTE)] for i, lbl in enumerate(unique_labels)}

    for i, ckpt_path in enumerate(checkpoints):
        display_name = f"{os.path.basename(os.path.dirname(ckpt_path))} | {os.path.basename(ckpt_path)}"
        print(f"\n[{i+1}/{len(checkpoints)}] {display_name}")

        # 使用新的智能加载函数
        model, device = load_model_smart(ckpt_path)
        if model is None: continue

        features_list = []
        labels_list = []

        # 1. 提取特征 (FP32)
        with torch.no_grad():
            for batch_imgs, batch_labels in tqdm(loader, desc="⚡ 特征提取"):
                batch_imgs = batch_imgs.to(device, non_blocking=True)
                feats = model(batch_imgs)
                if feats.ndim == 3: feats = feats.flatten(1)
                features_list.append(feats.cpu().numpy())
                labels_list.extend(batch_labels)

        if not features_list: continue
        X = np.concatenate(features_list, axis=0)
        y = np.array(labels_list)

        # 2. 清洗 NaN
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # 3. 方差筛选 (Threshold=0 剔除死特征)
        print(f"   🧹 原始维度: {X.shape[1]}")
        selector = VarianceThreshold(threshold=0)
        try:
            X = selector.fit_transform(X)
            print(f"   ✨ 有效维度: {X.shape[1]}")
        except ValueError:
            print("   ⚠️ 严重错误: 所有特征均为 0。请检查：1.输入数据是否正常 2.模型结构是否匹配(如CBAM)。")
            continue

        # 4. t-SNE
        print(f"   🧮 计算 3D t-SNE...")
        perp = min(30, len(X) - 1)
        tsne = TSNE(n_components=3, perplexity=perp, init='pca', learning_rate='auto', random_state=42)
        X_emb = tsne.fit_transform(X)

        save_base = f"3D_tSNE_{os.path.basename(os.path.dirname(ckpt_path))}_{os.path.basename(ckpt_path).replace('.pth','')}"

        # 5. 生成 PDF/SVG (可编辑)
        print("   🎨 渲染 PDF/SVG...")
        fig = plt.figure(figsize=(10, 8), dpi=300)
        ax = fig.add_subplot(111, projection='3d')

        for lbl in unique_labels:
            idx = y == lbl
            if np.sum(idx) > 0:
                ax.scatter(
                    X_emb[idx, 0], X_emb[idx, 1], X_emb[idx, 2],
                    c=label_to_color[lbl], label=lbl,
                    s=50, alpha=0.8, edgecolors='white', linewidth=0.2
                )

        ax.set_xlabel("Dim 1", fontweight='bold', labelpad=10)
        ax.set_ylabel("Dim 2", fontweight='bold', labelpad=10)
        ax.set_zlabel("Dim 3", fontweight='bold', labelpad=12) # 增加 padding 防止裁切
        ax.view_init(elev=30, azim=-60)
        ax.grid(False)
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        leg = ax.legend(loc='upper right', frameon=True, edgecolor='black', fancybox=False)
        leg.get_frame().set_linewidth(1.0)

        plt.savefig(os.path.join(OUTPUT_DIR, f"{save_base}.pdf"), format='pdf', bbox_inches='tight', pad_inches=0.5)
        plt.savefig(os.path.join(OUTPUT_DIR, f"{save_base}.svg"), format='svg', bbox_inches='tight', pad_inches=0.5)
        plt.close(fig)

        # 6. 生成 HTML (交互)
        if PLOTLY_AVAILABLE:
            plotly_data = []
            for lbl in unique_labels:
                idx = y == lbl
                if np.sum(idx) > 0:
                    plotly_data.append(go.Scatter3d(
                        x=X_emb[idx, 0], y=X_emb[idx, 1], z=X_emb[idx, 2],
                        mode='markers', name=lbl,
                        marker=dict(size=4, color=label_to_color[lbl], opacity=0.8, line=dict(width=0.2, color='white'))
                    ))
            fig_ply = go.Figure(data=plotly_data)
            fig_ply.update_layout(
                title=display_name,
                scene=dict(xaxis_title='Dim 1', yaxis_title='Dim 2', zaxis_title='Dim 3', bgcolor="white"),
                margin=dict(l=0, r=0, b=0, t=30)
            )
            fig_ply.write_html(os.path.join(OUTPUT_DIR, f"{save_base}_Interactive.html"))

        print(f"   ✅ 完成")

    print(f"\n🎉 全部任务结束. 结果在: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()