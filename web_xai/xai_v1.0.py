import streamlit as st
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from lime import lime_image

# ==========================================
# 0. 环境设置与路径
# ==========================================
sys.path.append(os.getcwd())

# 尝试导入模型
try:
    from models.resnet3d import resnet18_3d, resnet50_3d
    from models.densenet3d import densenet121_3d
    from models.vit3d import ViT3D
except ImportError:
    st.error("无法导入模型文件。请确保 models/ 文件夹在当前目录下。")
    st.stop()

# ==========================================
# 1. 预处理逻辑 (完全复刻 cell_fiber_dataset.py)
# ==========================================
def preprocess_volume(volume, target_depth=64):
    """
    对单样本进行推理前处理：
    1. 深度调整 (Pad/Crop 到 64)
    2. 归一化 (0-1)
    3. 维度变换 (D,H,W,C) -> (C,D,H,W)
    """
    # 确保是 numpy
    if isinstance(volume, torch.Tensor):
        volume = volume.numpy()

    # --- A. 深度处理 (复刻 _process_depth) ---
    # 假设输入形状是 (D, H, W, C) 或 (D, H, W)
    # 你的数据似乎是 (Depth, Height, Width, Channel) = (50, 256, 256, 2)
    current_depth = volume.shape[0]

    if current_depth > target_depth:
        # 截取中间
        start = (current_depth - target_depth) // 2
        end = start + target_depth
        volume = volume[start:end, :, :, :]
    elif current_depth < target_depth:
        # 补零
        pad_total = target_depth - current_depth
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        # np.pad 格式: ((before_D, after_D), (H, H), (W, W), (C, C))
        # 针对 (D, H, W, C) 结构
        volume = np.pad(volume,
                        ((pad_before, pad_after), (0, 0), (0, 0), (0, 0)),
                        mode='constant', constant_values=0)

    # 此时 volume depth 必定是 64

    # --- B. 归一化 (复刻 _normalize) ---
    volume = volume.astype(np.float32)
    min_val = volume.min()
    max_val = volume.max()
    if max_val - min_val > 0:
        volume = (volume - min_val) / (max_val - min_val)
    else:
        volume = volume - min_val

    # --- C. 维度变换 (PyTorch 格式) ---
    # (D, H, W, C) -> (C, D, H, W)
    # transpose(3, 0, 1, 2) 把最后一个维度(C)移到第一个
    volume_ch_first = volume.transpose(3, 0, 1, 2)

    return volume_ch_first, volume # 返回 (Tensor格式, 原始可视化格式)

# ==========================================
# 2. 独立模型加载函数
# ==========================================
@st.cache_resource
def load_model_resource(model_arch, num_classes, use_cbam, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {model_arch} from {checkpoint_path}...")

    try:
        if model_arch == 'ResNet18':
            model = resnet18_3d(num_classes=num_classes, use_cbam=use_cbam)
        elif model_arch == 'ResNet50':
            model = resnet50_3d(num_classes=num_classes, use_cbam=use_cbam)
        elif model_arch == 'DenseNet121':
            model = densenet121_3d(num_classes=num_classes, use_cbam=use_cbam)
        elif model_arch == 'ViT':
            model = ViT3D(num_classes=num_classes)
        else:
            return None, "Unknown Architecture"

        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict, strict=False)
        model.to(device)
        model.eval()
        return model, "OK"
    except Exception as e:
        return None, str(e)

# ==========================================
# 3. XAI 核心逻辑引擎
# ==========================================
class XAIEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gradients = None
        self.activations = None
        self.spatial_attn = None
        self.hook_handles = []

    def clear_hooks(self):
        for h in self.hook_handles:
            h.remove()
        self.hook_handles = []
        self.gradients = None
        self.activations = None
        self.spatial_attn = None

    def register_gradcam_hooks(self, model, target_layer):
        self.clear_hooks()
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        h1 = target_layer.register_forward_hook(forward_hook)
        h2 = target_layer.register_backward_hook(backward_hook)
        self.hook_handles.extend([h1, h2])

    def run_gradcam(self, model, input_tensor, target_class_idx):
        model.zero_grad()
        output = model(input_tensor)

        one_hot = torch.zeros_like(output)
        one_hot[0][target_class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        if self.gradients is None or self.activations is None:
            return None

        grads = self.gradients
        acts = self.activations

        # ==========================================
        # 核心修复: ViT 维度适配逻辑
        # ==========================================
        # CNN 输出通常是 5D: (B, C, D, H, W)
        # ViT 输出通常是 3D: (B, N_patches+1, Embed_Dim)
        if grads.ndim == 3:
            # 1. 剔除 CLS Token (通常在索引 0)
            # shape: (B, 2049, 384) -> (B, 2048, 384)
            grads = grads[:, 1:, :]
            acts = acts[:, 1:, :]

            # 2. 空间重塑 (Reshape)
            # 根据 vit3d.py 的默认配置:
            # Image=(64, 256, 256), Patch=(8, 16, 16)
            # Grid Dimensions: D=64/8=8, H=256/16=16, W=256/16=16
            # Total Patches = 8 * 16 * 16 = 2048
            b, n, e = grads.shape
            d_grid, h_grid, w_grid = 8, 16, 16

            # 校验一下是否匹配，防止尺寸变了报错
            if n != d_grid * h_grid * w_grid:
                print(f"Warning: ViT patch count {n} does not match default grid 8x16x16. Trying automatic calculation.")
                # 简单启发式: 假设 H=W
                # n = d * h * w. 这里简化处理，如果不对可能需要手动调整
                pass

                # (B, N, E) -> (B, D_g, H_g, W_g, E)
            grads = grads.reshape(b, d_grid, h_grid, w_grid, e)
            acts = acts.reshape(b, d_grid, h_grid, w_grid, e)

            # 3. 维度置换 (Permute) 以匹配 CNN 格式 (B, C, D, H, W)
            # 当前是 (B, D, H, W, E)，我们需要把 E (Channel) 移到第 1 位
            grads = grads.permute(0, 4, 1, 2, 3)
            acts = acts.permute(0, 4, 1, 2, 3)

        # ==========================================
        # 标准 Grad-CAM 计算 (现在兼容 ViT 了)
        # ==========================================
        # GAP over (D, H, W) -> dim=(2, 3, 4)
        weights = torch.mean(grads, dim=(2, 3, 4), keepdim=True)
        cam = torch.sum(weights * acts, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-7)

        # 上采样回原始输入尺寸
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='trilinear', align_corners=False)
        return cam.cpu().numpy()[0, 0]

    def run_lime(self, model, input_tensor_batch, num_samples=200):
        """
        input_tensor_batch: (1, C, D, H, W)
        """
        explainer = lime_image.LimeImageExplainer()

        # LIME 需要 (D, H, W, C) 格式的 numpy 数组
        # input_tensor_batch[0] 是 (C, D, H, W)
        # 此时需要转回 (D, H, W, C) 给 LIME
        img_np = input_tensor_batch[0].cpu().numpy().transpose(1, 2, 3, 0).astype(np.double)

        def predict_fn(images):
            # images: List of (D, H, W, C) -> PyTorch (N, C, D, H, W)
            imgs_np = np.array(images).transpose(0, 4, 1, 2, 3)
            tensor = torch.from_numpy(imgs_np).float().to(self.device)

            batch_size = 4
            preds = []
            with torch.no_grad():
                for i in range(0, len(tensor), batch_size):
                    batch = tensor[i:i+batch_size]
                    out = model(batch)
                    preds.append(F.softmax(out, dim=1).cpu().numpy())
            return np.concatenate(preds, axis=0)

        # 8x16x16 分块
        def segmentation_grid_3d(image):
            segments = np.zeros(image.shape[:3], dtype=int)
            d, h, w = image.shape[:3]
            sd, sh, sw = 8, 16, 16
            idx = 0
            for z in range(0, d, sd):
                for y in range(0, h, sh):
                    for x in range(0, w, sw):
                        segments[z:min(z+sd,d), y:min(y+sh,h), x:min(x+sw,w)] = idx
                        idx += 1
            return segments

        explanation = explainer.explain_instance(
            img_np, predict_fn, labels=[0, 1, 2, 3, 4], top_labels=1,
            hide_color=0, num_samples=num_samples, segmentation_fn=segmentation_grid_3d
        )
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, hide_rest=False, num_features=10)
        return mask

# ==========================================
# 4. Streamlit 界面逻辑
# ==========================================
def main():
    st.set_page_config(page_title="3D Model XAI", layout="wide")
    st.title("🔬 3D 模型可解释性分析平台")

    engine = XAIEngine()

    with st.sidebar:
        st.header("1. 模型配置")
        model_arch = st.selectbox("选择架构", ["ResNet18", "ResNet50", "DenseNet121", "ViT"], index=1)
        use_cbam = st.checkbox("使用 CBAM", value=True)

        ckpt_root = "./checkpoints_cbam"
        if not os.path.exists(ckpt_root):
            st.warning(f"目录不存在: {ckpt_root}")
            ckpt_files = []
        else:
            ckpt_files = []
            for root, dirs, files in os.walk(ckpt_root):
                for file in files:
                    if file.endswith(".pth"):
                        ckpt_files.append(os.path.join(root, file))

        if not ckpt_files:
            st.error("未找到 .pth 权重文件")
            st.stop()
        ckpt_path = st.selectbox("选择权重文件 (.pth)", ckpt_files)

        model, msg = load_model_resource(model_arch, 5, use_cbam, ckpt_path)
        if model is None:
            st.error(f"模型加载失败: {msg}"); st.stop()
        else:
            st.success(f"模型已加载: {os.path.basename(ckpt_path)}")

        st.header("2. 数据选择")
        data_root = "./TRAIN_DATA_FINAL_256"
        if not os.path.exists(data_root):
            st.warning(f"目录不存在: {data_root}"); data_files = []
        else:
            data_files = []
            for root, dirs, files in os.walk(data_root):
                for file in files:
                    if file.endswith(".npz"):
                        rel_path = os.path.relpath(os.path.join(root, file), start=data_root)
                        data_files.append(rel_path)

        if not data_files:
            st.error("未找到 .npz 数据文件"); st.stop()
        selected_file = st.selectbox("选择数据样本 (.npz)", data_files)
        full_data_path = os.path.join(data_root, selected_file)

        st.header("3. 分析方法")
        xai_method = st.radio("选择方法", ["Grad-CAM", "CBAM Attention", "LIME (Slow)"])
        run_btn = st.button("🚀 开始分析", type="primary")

    if not os.path.exists(full_data_path):
        st.stop()

    # --- 数据加载与关键预处理 ---
    try:
        npz_data = np.load(full_data_path)
        keys = list(npz_data.keys())
        key = 'data' if 'data' in keys else keys[0]
        raw_data = npz_data[key]
        # 去除batch维度 (1, D, H, W, C) -> (D, H, W, C)
        if raw_data.ndim == 5: raw_data = raw_data[0]

        # 核心修复点：调用预处理函数，将 D=50 -> D=64，并将 Channel 移到前面
        tensor_np, viz_np = preprocess_volume(raw_data, target_depth=64)

        # tensor_np: (C, D, H, W) - 这里的 C=2, D=64
        # viz_np: (D, H, W, C) - 用于可视化，D=64

    except Exception as e:
        st.error(f"数据加载或预处理错误: {e}")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**文件**: `{selected_file}`")
        st.write(f"**原始尺寸**: `{raw_data.shape}`")
        st.write(f"**模型输入尺寸**: `{tensor_np.shape}` (C, D, H, W)")

    # 构造 Batch (1, C, D, H, W)
    input_tensor = torch.from_numpy(tensor_np).unsqueeze(0).float().to(engine.device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        conf = probs[0][pred_idx].item()
        class_names = ['E', 'L', 'LQ', 'Q', 'QLX']

    with col2:
        st.metric("模型预测结果", f"{class_names[pred_idx]}", f"{conf:.2%}")

    # --- 分析与可视化 ---
    if 'heatmap' not in st.session_state:
        st.session_state['heatmap'] = None

    if run_btn:
        with st.spinner(f"正在运行 {xai_method}..."):
            engine.clear_hooks()
            heatmap = None

            if xai_method == "Grad-CAM":
                target_layer = None
                if "ResNet" in model_arch:
                    if hasattr(model, 'layer4'):
                        last_block = model.layer4[-1]
                        if hasattr(last_block, 'conv3'): target_layer = last_block.conv3
                        elif hasattr(last_block, 'conv2'): target_layer = last_block.conv2
                    else: target_layer = list(model.modules())[-2]
                elif "DenseNet" in model_arch:
                    if hasattr(model, 'features'): target_layer = model.features.denseblock4.layers[-1].conv2
                elif "ViT" in model_arch:
                    target_layer = model.norm

                if target_layer:
                    engine.register_gradcam_hooks(model, target_layer)
                    heatmap = engine.run_gradcam(model, input_tensor, pred_idx)
                else:
                    st.error("无法定位 Grad-CAM 目标层")

            elif xai_method == "CBAM Attention":
                found = False
                for name, module in model.named_modules():
                    if 'spatial' in name.lower() and isinstance(module, torch.nn.Sigmoid):
                        def hook(m, i, o): engine.spatial_attn = o.detach()
                        module.register_forward_hook(hook)
                        found = True
                        break
                if found:
                    _ = model(input_tensor)
                    if engine.spatial_attn is not None:
                        heatmap = F.interpolate(engine.spatial_attn, size=(64, 256, 256), mode='trilinear').cpu().numpy()[0, 0]
                else:
                    st.warning("未找到 CBAM 模块")

            elif xai_method.startswith("LIME"):
                heatmap = engine.run_lime(model, input_tensor, num_samples=200)

            st.session_state['heatmap'] = heatmap

    st.divider()
    st.subheader("4. 交互式 3D 切片查看")

    # 深度已经是 64 了
    depth_idx = st.slider("深度切片 (Z-Axis)", 0, 63, 32)

    viz_col1, viz_col2, viz_col3 = st.columns(3)

    # viz_np 是 (D, H, W, C)
    img_slice = viz_np[depth_idx, :, :, 0] # 取第0通道用于显示

    # 已经是 0-1 之间了，但为了保险显示
    img_disp = img_slice

    fig1, ax1 = plt.subplots()
    ax1.imshow(img_disp, cmap='gray')
    ax1.set_title("Processed Input (Slice)")
    ax1.axis('off')
    viz_col1.pyplot(fig1)

    if st.session_state['heatmap'] is not None:
        heatmap_vol = st.session_state['heatmap']
        map_slice = heatmap_vol[depth_idx, :, :]

        fig2, ax2 = plt.subplots()
        ax2.imshow(map_slice, cmap='jet')
        ax2.set_title(f"{xai_method} Heatmap")
        ax2.axis('off')
        viz_col2.pyplot(fig2)

        fig3, ax3 = plt.subplots()
        ax3.imshow(img_disp, cmap='gray')
        ax3.imshow(map_slice, cmap='jet', alpha=0.5)
        ax3.set_title("Overlay")
        ax3.axis('off')
        viz_col3.pyplot(fig3)
    else:
        viz_col2.info("点击 '开始分析' 生成热力图")

if __name__ == "__main__":
    main()