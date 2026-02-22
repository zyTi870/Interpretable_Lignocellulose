import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

# --- 引入您的模型 ---
from models.resnet3d import resnet18_3d, resnet50_3d
from models.densenet3d import densenet121_3d
from models.vit3d import ViT3D
from history.cell_fiber_dataset import CellFiberCleanDataset

# --- 配置区域 ---
CONFIG = {
    'model_type': 'resnet50_cbam',
    'checkpoint_path': './checkpoints_cbam/resnet50_cbam/resnet50_cbam_best.pth',
    'sample_path': './TRAIN_DATA_FINAL_256/E/E0102_fixed_BL.npz',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_classes': 5,
    'target_layer_name': 'layer3',
}

CLASS_MAPPING = {'E': 0, 'L': 1, 'LQ': 2, 'Q': 3, 'QLX': 4}

# --- 1. 模型加载 ---
def load_model(config):
    print(f"Loading model: {config['model_type']}...")
    if config['model_type'] == 'resnet50_cbam':
        model = resnet50_3d(num_classes=config['num_classes'], use_cbam=True)
    elif config['model_type'] == 'resnet18_cbam':
        model = resnet18_3d(num_classes=config['num_classes'], use_cbam=True)
    elif config['model_type'] == 'densenet121':
        model = densenet121_3d(num_classes=config['num_classes'])
    else:
        model = ViT3D(num_classes=config['num_classes'])

    checkpoint = torch.load(config['checkpoint_path'], map_location=config['device'])
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    model.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()}, strict=False)
    model.to(config['device']).eval()
    return model

# --- 2. Grad-CAM ---
class GradCAM3D:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x):
        self.model.zero_grad()
        output = self.model(x)
        idx = torch.argmax(output, dim=1).item()
        one_hot = torch.zeros_like(output)
        one_hot[0][idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        grads = torch.mean(self.gradients, dim=[2, 3, 4])
        activs = self.activations.detach()
        for i in range(activs.shape[1]):
            activs[:, i, :, :, :] *= grads[:, i].view(-1, 1, 1, 1)

        heatmap = torch.mean(activs, dim=1).squeeze()
        return F.relu(heatmap).cpu().numpy(), idx

# --- 3. 渲染核心 (修复比例问题) ---
def render_perfect_cube(volume_ch, heatmap, title, save_name):
    print(f"\nProcessing {title}...")

    # 1. 智能 ROI 检测
    coords = np.argwhere(volume_ch > 0.05)
    if len(coords) == 0:
        print(f"Warning: Channel {title} is empty. Skipping.")
        return

    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0)

    # Padding
    pad = 2
    z_min = max(0, z_min - pad); z_max = min(volume_ch.shape[0], z_max + pad)
    y_min = max(0, y_min - pad); y_max = min(volume_ch.shape[1], y_max + pad)
    x_min = max(0, x_min - pad); x_max = min(volume_ch.shape[2], x_max + pad)

    print(f"ROI Detected: Z[{z_min}:{z_max}], Y[{y_min}:{y_max}], X[{x_min}:{x_max}]")

    vol_crop = volume_ch[z_min:z_max, y_min:y_max, x_min:x_max]
    hm_crop = heatmap[z_min:z_max, y_min:y_max, x_min:x_max]

    # 2. 降采样 (Isotropic Downsampling)
    # 我们使用统一的 scale_factor 缩放所有轴，保持物理比例不变
    max_dim = max(vol_crop.shape)
    # 限制最大边长为 64，防止显存爆炸
    scale_factor = 64.0 / max_dim if max_dim > 64 else 1.0

    if scale_factor < 1.0:
        vol_small = zoom(vol_crop, scale_factor, order=1)
        hm_small = zoom(hm_crop, scale_factor, order=1)
    else:
        vol_small = vol_crop
        hm_small = hm_crop

    D, H, W = vol_small.shape
    print(f"Rendering Grid: {D}x{H}x{W}")

    # 3. 定义掩码
    x, y, z = np.indices((D, H, W))

    # 剖切掩码 (切掉右上前方 1/4)
    cut_mask = (x >= D//2) & (y >= H//2)

    structure_mask = vol_small > 0.1
    heatmap_mask = hm_small > 0.2

    visible_mask = (structure_mask | heatmap_mask) & (~cut_mask)

    # 4. 颜色填充 (Safe Logic)
    colors = np.zeros(visible_mask.shape + (4,))

    vol_norm = (vol_small - vol_small.min()) / (vol_small.max() - vol_small.min() + 1e-8)
    hm_norm = (hm_small - hm_small.min()) / (hm_small.max() - hm_small.min() + 1e-8)

    if "Channel 0" in title:
        base_cmap = plt.get_cmap('bone')
        alpha_base = 0.1
    else:
        base_cmap = plt.get_cmap('viridis')
        alpha_base = 0.15

    jet_cmap = plt.get_cmap('jet')

    # 填充结构
    struct_fill_mask = structure_mask & (~cut_mask)
    if np.any(struct_fill_mask):
        rgba = base_cmap(vol_norm[struct_fill_mask])
        rgba[:, 3] = alpha_base
        colors[struct_fill_mask] = rgba

    # 填充热图
    hm_fill_mask = heatmap_mask & (~cut_mask)
    if np.any(hm_fill_mask):
        rgba = jet_cmap(hm_norm[hm_fill_mask])
        rgba[:, 3] = 0.9
        colors[hm_fill_mask] = rgba

    # 5. 绘图与比例修复
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.voxels(visible_mask, facecolors=colors, edgecolor=None, shade=True)

    # --- 关键修改：锁定轴比例 ---
    # 这行代码告诉 matplotlib：Z轴的长度显示应与数据中的 D 成正比
    # 这样 D 如果只有 10，看起来就是薄薄的一层，像素会变回正方体
    ax.set_box_aspect((D, H, W))

    ax.view_init(elev=30, azim=45)
    ax.axis('off')

    plt.title(f"{title}\n(Correct Aspect Ratio)", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_name}")

# --- 主逻辑 ---
def main():
    print("Loading data...")
    ds = CellFiberCleanDataset([CONFIG['sample_path']], target_depth=64, class_mapping=CLASS_MAPPING)
    img_tensor, label = ds[0]
    img_input = img_tensor.unsqueeze(0).to(CONFIG['device'])

    model = load_model(CONFIG)

    print("Computing Grad-CAM...")
    if 'resnet' in CONFIG['model_type']:
        target_layer = model.layer3[-1]
    else:
        target_layer = list(model.modules())[-2]

    cam = GradCAM3D(model, target_layer)
    heatmap_raw, pred_idx = cam(img_input)

    print("Upsampling heatmap...")
    hm_tensor = torch.tensor(heatmap_raw).unsqueeze(0).unsqueeze(0)
    target_shape = (64, 256, 256)
    hm_up = F.interpolate(hm_tensor, size=target_shape, mode='trilinear', align_corners=False)
    heatmap_final = hm_up.squeeze().numpy()
    heatmap_final = (heatmap_final - heatmap_final.min()) / (heatmap_final.max() - heatmap_final.min() + 1e-8)

    vol_ch0 = img_tensor[0].cpu().numpy()
    vol_ch1 = img_tensor[1].cpu().numpy()

    # 生成两张图
    render_perfect_cube(vol_ch0, heatmap_final, "Channel 0 (Structure) + GradCAM", "vis_ch0_final.png")
    render_perfect_cube(vol_ch1, heatmap_final, "Channel 1 (Fiber) + GradCAM", "vis_ch1_final.png")

    print("\nSuccess! Generated 'vis_ch0_final.png' and 'vis_ch1_final.png'.")

if __name__ == '__main__':
    main()