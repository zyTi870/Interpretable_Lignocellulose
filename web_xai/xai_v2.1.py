import streamlit as st
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from datetime import datetime

# ==========================================
# 0. 基础配置
# ==========================================
st.set_page_config(layout="wide", page_title="CNN Fiber 3D XAI v5.3 (Fixed Syntax)")
sys.path.append(os.getcwd())

# CSS 优化
st.markdown("""
<style>
    div.stButton > button {
        width: 100%;
        font-weight: bold;
        border-radius: 8px;
        height: 3em;
    }
    .block-container {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# 字体设置
try:
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'stix'
except:
    pass

# 输出目录
ROOT_OUTPUT_DIR = "../3d_processing_outputs"
DIRS = {
    "screenshots": os.path.join(ROOT_OUTPUT_DIR, "output_3d_snapshots"),
    "slices": os.path.join(ROOT_OUTPUT_DIR, "output_2d_slices"),
    "npz": os.path.join(ROOT_OUTPUT_DIR, "output_processed_npz")
}
for d in DIRS.values(): os.makedirs(d, exist_ok=True)

# ==========================================
# 1. 核心工具函数
# ==========================================
def dynamic_rescaling(tensor_or_array):
    if isinstance(tensor_or_array, torch.Tensor):
        arr = tensor_or_array.detach().cpu().numpy()
    else:
        arr = tensor_or_array.copy()
    if arr.max() == arr.min():
        return np.zeros_like(arr)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

def preprocess_volume(volume, target_depth=64):
    if isinstance(volume, torch.Tensor): volume = volume.numpy()
    volume = volume.astype(np.float32)
    volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
    current_depth = volume.shape[0]
    if current_depth > target_depth:
        start = (current_depth - target_depth) // 2
        volume = volume[start:start+target_depth, :, :, :]
    elif current_depth < target_depth:
        pad = target_depth - current_depth
        volume = np.pad(volume, ((pad//2, pad - pad//2), (0,0), (0,0), (0,0)), mode='constant')
    return volume.transpose(3, 0, 1, 2), volume

def add_scale_bar_2d(ax, image_width_px, pixel_size_um=1.32, bar_length_um=50):
    bar_pixels = bar_length_um / pixel_size_um
    H, W = image_width_px
    x0 = W - bar_pixels - 15
    y0 = H - 15
    rect = Rectangle((x0, y0), bar_pixels, H*0.015, color='white', linewidth=0)
    ax.add_patch(rect)
    ax.text(x0 + bar_pixels/2, y0 - 10, f'{bar_length_um} $\mu m$',
            color='white', ha='center', va='bottom', fontsize=10, fontweight='bold', fontname='Times New Roman')

# ==========================================
# 2. 模型与 XAI 引擎
# ==========================================
try:
    from models.resnet3d import resnet18_3d, resnet50_3d, SpatialAttention3D
    from models.densenet3d import densenet121_3d
    from models.vit3d import ViT3D
except ImportError:
    pass

@st.cache_resource
def load_model_resource(model_arch, num_classes, use_cbam, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        if model_arch == 'ResNet18': model = resnet18_3d(num_classes=num_classes, use_cbam=use_cbam)
        elif model_arch == 'ResNet50': model = resnet50_3d(num_classes=num_classes, use_cbam=use_cbam)
        elif model_arch == 'DenseNet121': model = densenet121_3d(num_classes=num_classes, use_cbam=use_cbam)
        elif model_arch == 'ViT': model = ViT3D(num_classes=num_classes)
        else: return None, "Unknown Architecture"

        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
        model.to(device)
        model.eval()
        return model, "OK"
    except Exception as e: return None, str(e)

class UltimateXAIEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hooks = []
        self.data = None
    def clear(self):
        for h in self.hooks: h.remove()
        self.hooks = []
        self.data = None
    def _resize_and_norm(self, tensor, target_size):
        tensor = F.interpolate(tensor, size=target_size, mode='trilinear', align_corners=True)
        return dynamic_rescaling(tensor[0, 0])

    def run_auto(self, model, x, method, arch_name, pred_class):
        self.clear()
        if method == "Grad-CAM":
            target_layer = None
            if "ResNet" in arch_name:
                block = model.layer4
                target_layer = block[-1].conv3 if hasattr(block[-1], 'conv3') else block[-1].conv2
            elif "DenseNet" in arch_name:
                target_layer = model.features.denseblock4

            if target_layer:
                grads, acts = {}, {}
                self.hooks.append(target_layer.register_forward_hook(lambda m,i,o: acts.update({'val': o.detach()})))
                self.hooks.append(target_layer.register_full_backward_hook(lambda m,gi,go: grads.update({'val': go[0].detach()})))
                model.zero_grad()
                out = model(x)
                one_hot = torch.zeros_like(out); one_hot[0][pred_class] = 1
                out.backward(gradient=one_hot, retain_graph=True)
                g, a = grads.get('val'), acts.get('val')
                if g is not None and a is not None:
                    if g.ndim == 3:
                        g = g[:, 1:]; a = a[:, 1:]
                        b, n, e = g.shape; d, h, w = 8, 16, 16
                        g = g.reshape(b, d, h, w, e).permute(0, 4, 1, 2, 3)
                        a = a.reshape(b, d, h, w, e).permute(0, 4, 1, 2, 3)
                    weights = torch.mean(g, dim=(2,3,4), keepdim=True)
                    cam = torch.sum(weights * a, dim=1, keepdim=True)
                    return self._resize_and_norm(F.relu(cam), x.shape[2:])

        elif method == "CBAM Attention":
            target = None
            for m in model.modules():
                if isinstance(m, SpatialAttention3D): target = m
            if target:
                self.hooks.append(target.register_forward_hook(lambda m,i,o: setattr(self, 'data', o.detach())))
                with torch.no_grad(): model(x)
                return self._resize_and_norm(self.data, x.shape[2:])

        elif method == "ViT Attention":
            target = model.blocks[-1].attn.attn_drop if hasattr(model, 'blocks') else None
            if target:
                self.hooks.append(target.register_forward_hook(lambda m,i,o: setattr(self, 'data', i[0].detach())))
                with torch.no_grad(): model(x)
                attn = torch.mean(self.data, dim=1)[:, 0, 1:]
                b, n = attn.shape; attn = attn.reshape(b, 1, 8, 16, 16)
                return self._resize_and_norm(attn, x.shape[2:])
        return None

    def run_gradcam(self, model, x, target_layer, target_class):
        self.clear()
        grads, acts = {}, {}
        self.hooks.append(target_layer.register_forward_hook(lambda m,i,o: acts.update({'val': o.detach()})))
        self.hooks.append(target_layer.register_full_backward_hook(lambda m,gi,go: grads.update({'val': go[0].detach()})))
        model.zero_grad()
        out = model(x)
        one_hot = torch.zeros_like(out); one_hot[0][target_class] = 1
        out.backward(gradient=one_hot, retain_graph=True)
        g, a = grads.get('val'), acts.get('val')
        if g is None or a is None: return None
        if g.ndim == 3:
            g = g[:, 1:]; a = a[:, 1:]
            b, n, e = g.shape; d, h, w = 8, 16, 16
            g = g.reshape(b, d, h, w, e).permute(0, 4, 1, 2, 3)
            a = a.reshape(b, d, h, w, e).permute(0, 4, 1, 2, 3)
        weights = torch.mean(g, dim=(2,3,4), keepdim=True)
        cam = torch.sum(weights * a, dim=1, keepdim=True)
        return self._resize_and_norm(F.relu(cam), x.shape[2:])

# ==========================================
# 3. 渲染引擎
# ==========================================

# 红绿双通道点云 (用于预览)
def generate_rg_point_cloud(volume, max_points=60000, threshold=0.1):
    D, H, W, C = volume.shape
    z, y, x = np.indices((D, H, W))

    # 获取两个通道的数据
    v0 = dynamic_rescaling(volume[..., 0]) # Red
    v1 = dynamic_rescaling(volume[..., 1]) if C > 1 else np.zeros_like(v0) # Green

    # 联合掩码
    mask = (v0 > threshold) | (v1 > threshold)

    pts_x = x[mask]; pts_y = y[mask]; pts_z = z[mask]
    pts_v0 = v0[mask]
    pts_v1 = v1[mask]

    # 采样
    num_points = len(pts_x)
    if num_points > max_points:
        idx = np.random.choice(num_points, max_points, replace=False)
        pts_x = pts_x[idx]; pts_y = pts_y[idx]; pts_z = pts_z[idx]
        pts_v0 = pts_v0[idx]; pts_v1 = pts_v1[idx]

    # 生成颜色字符串 list: rgb(r, g, 0)
    r_vals = (pts_v0 * 255).astype(int)
    g_vals = (pts_v1 * 255).astype(int)
    color_strings = [f'rgb({r},{g},0)' for r, g in zip(r_vals, g_vals)]

    return pts_x, pts_y, pts_z, color_strings

def generate_point_cloud(volume, heatmap, roi_bounds=None, thresholds=None, max_points=100000):
    D, H, W = volume.shape
    z, y, x = np.indices((D, H, W))
    vol_norm = dynamic_rescaling(volume)

    if roi_bounds is None: roi_bounds = ((0,D), (0,H), (0,W))
    if thresholds is None: thresholds = {'structure': 0.1}

    (z_min, z_max), (y_min, y_max), (x_min, x_max) = roi_bounds
    roi_mask = ((z >= z_min) & (z < z_max) & (y >= y_min) & (y < y_max) & (x >= x_min) & (x < x_max))
    struct_mask = vol_norm > thresholds['structure']
    final_mask = roi_mask & struct_mask

    pts_x = x[final_mask]; pts_y = y[final_mask]; pts_z = z[final_mask]
    pts_v = vol_norm[final_mask]

    if heatmap is not None:
        hm_norm = dynamic_rescaling(heatmap)
        pts_h = hm_norm[final_mask]
    else:
        pts_h = None

    num_points = len(pts_x)
    if num_points > max_points:
        if pts_h is not None:
            weights = pts_h + 0.1
            weights /= weights.sum()
            idx = np.random.choice(num_points, max_points, replace=False, p=weights)
        else:
            idx = np.random.choice(num_points, max_points, replace=False)
        pts_x = pts_x[idx]; pts_y = pts_y[idx]; pts_z = pts_z[idx]
        pts_v = pts_v[idx]
        if pts_h is not None: pts_h = pts_h[idx]

    return pts_x, pts_y, pts_z, pts_v, pts_h

def render_plotly_figure_v5(pts_data, title, camera_eye, volume_shape, is_preview=False, is_rg=False):
    # is_rg: 是否为红绿双通道模式
    if is_rg:
        px, py, pz, color_strings = pts_data
        marker_cfg = dict(
            size=2,
            color=color_strings,
            opacity=0.3,
            showscale=False
        )
    else:
        px, py, pz, pv, ph = pts_data
        if ph is not None:
            color_val = ph; colorscale = 'Jet'; opacity = 0.3
        else:
            color_val = pv; colorscale = 'gray'; opacity = 0.15

        marker_cfg = dict(
            size=2, color=color_val, colorscale=colorscale, opacity=opacity,
            colorbar=dict(title="Intensity") if not is_preview else None,
            showscale=not is_preview
        )

    data = [go.Scatter3d(
        x=px, y=py, z=pz, mode='markers',
        marker=marker_cfg,
        name=title,
        hoverinfo='skip' if is_preview else 'all'
    )]

    if not is_preview:
        D, H, W = volume_shape
        scale_um = 50; pixel_size = 1.32; bar_len_px = scale_um / pixel_size
        offset = 10
        x_bar = [W - offset - bar_len_px, W - offset]
        y_bar = [H - offset, H - offset]
        z_bar = [0, 0]

        data.append(go.Scatter3d(
            x=x_bar, y=y_bar, z=z_bar, mode='lines+text',
            line=dict(color='black', width=3),
            text=[f"{scale_um} um", ""], textposition="top center",
            textfont=dict(color='black', size=14, family='Times New Roman'),
            showlegend=False
        ))

    axis_config = dict(
        showgrid=True, gridcolor='rgba(200, 200, 200, 0.5)',
        showline=True, linecolor='black', linewidth=2,
        backgroundcolor="rgba(255,255,255,0)",
        visible=not is_preview
    )

    layout = go.Layout(
        title=title, autosize=True,
        height=400 if is_preview else 600,
        scene=dict(
            xaxis=dict(title='X', **axis_config),
            yaxis=dict(title='Y', **axis_config),
            zaxis=dict(title='Z', **axis_config),
            aspectmode='data',
            camera=dict(eye=camera_eye),
            dragmode='turntable'
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        paper_bgcolor="rgba(255, 255, 255, 0)"
    )
    return go.Figure(data=data, layout=layout)

# 红绿双通道 2D 切片保存与绘制
def save_rg_slice(vol_slice_2ch, hm_slice, title_str, save_name):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    H, W, C = vol_slice_2ch.shape

    # 构建 RGB 底图
    rgb_img = np.zeros((H, W, 3), dtype=np.float32)
    v0 = dynamic_rescaling(vol_slice_2ch[..., 0]) # R
    v1 = dynamic_rescaling(vol_slice_2ch[..., 1]) if C > 1 else np.zeros_like(v0) # G

    rgb_img[..., 0] = v0
    rgb_img[..., 1] = v1
    # B remains 0

    ax.imshow(rgb_img, alpha=1.0)
    if hm_slice is not None:
        h_img = dynamic_rescaling(hm_slice)
        h_masked = np.ma.masked_where(h_img < 0.2, h_img)
        ax.imshow(h_masked, cmap='jet', alpha=0.5)

    ax.axis('off')
    try: ax.set_title(title_str, fontname='Times New Roman', fontsize=14, pad=10)
    except: ax.set_title(title_str, fontsize=14, pad=10)

    add_scale_bar_2d(ax, (H, W), pixel_size_um=1.32, bar_length_um=50)

    path = os.path.join(DIRS["slices"], save_name)
    plt.savefig(path, bbox_inches='tight', pad_inches=0.05, dpi=300)
    plt.close(fig)
    return path

# ==========================================
# 4. 状态管理
# ==========================================
if 'page' not in st.session_state: st.session_state.page = 'browser'
if 'file_index' not in st.session_state: st.session_state.file_index = 0
if 'current_data' not in st.session_state: st.session_state.current_data = None
if 'heatmap' not in st.session_state: st.session_state.heatmap = None
if 'model_loaded' not in st.session_state: st.session_state.model_loaded = None

# ==========================================
# 5. 主程序
# ==========================================
def main():
    # --- Sidebar 配置 ---
    with st.sidebar:
        st.header("1. Model Setup")
        arch = st.selectbox("Architecture", ["ResNet50", "ResNet18", "DenseNet121", "ViT"])
        use_cbam = st.checkbox("Enable CBAM", True)

        ckpt_dir = "./checkpoints_cbam" if os.path.exists("../checkpoints_cbam") else "."
        ckpts = sorted([os.path.join(r, f) for r, d, fl in os.walk(ckpt_dir) for f in fl if f.endswith(".pth")])
        ckpt = st.selectbox("Checkpoint", ckpts, format_func=os.path.basename) if ckpts else None

        data_dir = "./TRAIN_DATA_FINAL_256" if os.path.exists("../TRAIN_DATA_FINAL_256") else "."
        all_files = sorted([os.path.join(r, f) for r, d, fl in os.walk(data_dir) for f in fl if f.endswith(".npz")])
        if not all_files: st.error("No Data"); st.stop()

        # 加载模型
        if st.session_state.model_loaded != (arch, ckpt):
            model, msg = load_model_resource(arch, 5, use_cbam, ckpt)
            if model:
                st.session_state.model = model
                st.session_state.model_loaded = (arch, ckpt)
                st.success(f"Loaded: {arch}")
            else:
                st.error(msg); st.stop()

    # === 页面 1: 浏览预览模式 ===
    if st.session_state.page == 'browser':
        st.title("📂 3D Browser & RG-XAI Preview")

        # 导航栏
        col_nav1, col_nav2, col_nav3 = st.columns([1, 4, 1])
        with col_nav1:
            if st.button("⬅️ PREV"):
                st.session_state.file_index = max(0, st.session_state.file_index - 1)
                st.rerun()

        with col_nav2:
            def format_func(i):
                return f"{i+1}. {os.path.basename(all_files[i])}"

            selected_idx = st.selectbox(
                "Jump to Sample:",
                range(len(all_files)),
                index=st.session_state.file_index,
                format_func=format_func,
                label_visibility="collapsed"
            )
            if selected_idx != st.session_state.file_index:
                st.session_state.file_index = selected_idx
                st.rerun()

        with col_nav3:
            if st.button("NEXT ➡️"):
                st.session_state.file_index = min(len(all_files)-1, st.session_state.file_index + 1)
                st.rerun()

        # 逻辑处理
        model = st.session_state.model
        f_path = all_files[st.session_state.file_index]

        # 1. 自动加载数据和计算默认XAI
        cache_key = f"{f_path}_{arch}"
        if 'preview_cache_key' not in st.session_state or st.session_state.preview_cache_key != cache_key:
            with st.spinner("🚀 GPU Rendering 3D Preview..."):
                with np.load(f_path) as f: raw = f['data'] if 'data' in f else f[list(f.keys())[0]]
                if raw.ndim == 5: raw = raw[0]
                tensor_in, vol_vis = preprocess_volume(raw)

                engine = UltimateXAIEngine()
                x = torch.from_numpy(tensor_in).unsqueeze(0).float().to(engine.device)

                # 预测
                logits = model(x)
                probs = F.softmax(logits, dim=1)
                pred_class = probs.argmax().item()

                # 默认计算 Grad-CAM
                hm = engine.run_auto(model, x, "Grad-CAM", arch, pred_class)

                # 更新状态
                st.session_state.current_data = (tensor_in, vol_vis)
                st.session_state.heatmap = hm
                st.session_state.pred_info = (pred_class, probs[0][pred_class].item())
                st.session_state.preview_cache_key = cache_key

        tensor_in, vol_vis = st.session_state.current_data
        heatmap = st.session_state.heatmap
        pred_cls, pred_conf = st.session_state.pred_info
        D, H, W, C = vol_vis.shape

        st.markdown(f"**Current File:** `{os.path.basename(f_path)}` | **Pred:** `Class {pred_cls}` ({pred_conf*100:.1f}%)")

        col_p1, col_p2 = st.columns(2)
        prev_pts = 60000; prev_th = 0.1; eye = dict(x=1.6, y=1.6, z=1.6)

        with col_p1:
            st.markdown("##### 🧊 Raw Structure (Red/Green Dual)")
            if C >= 1:
                pts_rg = generate_rg_point_cloud(vol_vis, max_points=prev_pts, threshold=prev_th)
                fig_rg = render_plotly_figure_v5(pts_rg, "RG Structure", eye, (D,H,W), is_preview=True, is_rg=True)
                st.plotly_chart(fig_rg, use_container_width=True)
            else:
                st.error("Data has less than 1 channel")

        with col_p2:
            st.markdown("##### 🔥 XAI Heatmap (Grad-CAM)")
            if heatmap is not None:
                pts_xai = generate_point_cloud(vol_vis[..., 0], heatmap, thresholds={'structure': prev_th}, max_points=prev_pts)
                fig_xai = render_plotly_figure_v5(pts_xai, "Attention", eye, (D,H,W), is_preview=True)
                st.plotly_chart(fig_xai, use_container_width=True)
            else:
                st.warning("Heatmap calculation failed.")

        st.divider()
        if st.button("🔬 Enter Detail Analysis (Adjustment & Export)", type="primary"):
            st.session_state.page = 'details'
            st.rerun()

    # === 页面 2: 详细分析 (保持不变) ===
    elif st.session_state.page == 'details':
        if st.button("⬅️ Back to Browser"):
            st.session_state.page = 'browser'
            st.rerun()

        st.title("🔬 Detailed Analysis Lab")

        f_path = all_files[st.session_state.file_index]
        sample_name = os.path.splitext(os.path.basename(f_path))[0]

        tensor_in, vol_vis = st.session_state.current_data
        model = st.session_state.model
        D, H, W, C = vol_vis.shape
        engine = UltimateXAIEngine()
        x = torch.from_numpy(tensor_in).unsqueeze(0).float().to(engine.device)

        tab_calc, tab_vis, tab_exp = st.tabs(["📊 Re-Calculate", "🖼️ 3D Adjustment", "💾 Save & Export"])

        with tab_calc:
            c1, c2 = st.columns([1, 2])
            with c1: method = st.radio("Method", ["Grad-CAM", "CBAM Attention", "ViT Attention"])
            with c2:
                layer_name = None
                if method == "Grad-CAM":
                    if "ResNet" in arch: layer_name = st.selectbox("Layer", ["layer4", "layer3", "layer2"])
                    elif "DenseNet" in arch: layer_name = st.selectbox("Block", ["denseblock4", "denseblock3"])

                if st.button("🚀 Calculate Custom Map"):
                    pred_cls = st.session_state.pred_info[0]
                    hm = None
                    if method == "Grad-CAM":
                        target_layer = None
                        if "ResNet" in arch:
                            block = getattr(model, layer_name)
                            target_layer = block[-1].conv3 if hasattr(block[-1], 'conv3') else block[-1].conv2
                        elif "DenseNet" in arch:
                            for n, m in model.features.named_children():
                                if n == layer_name: target_layer = m
                        if target_layer: hm = engine.run_gradcam(model, x, target_layer, pred_cls)
                    elif method == "CBAM Attention": hm, _ = engine.run_cbam(model, x)
                    elif method == "ViT Attention": hm, _ = engine.run_vit_attn(model, x)

                    if hm is not None:
                        st.session_state.heatmap = hm
                        st.success("Heatmap Updated!")

        with tab_vis:
            heatmap = st.session_state.heatmap
            if heatmap is None: st.warning("No heatmap available."); st.stop()

            with st.expander("Rendering Controls", expanded=True):
                c_roi, c_style = st.columns(2)
                with c_roi:
                    z_r = st.slider("Z Clip", 0, D, (0, D)); y_r = st.slider("Y Clip", 0, H, (0, H)); x_r = st.slider("X Clip", 0, W, (0, W))
                with c_style:
                    th = st.slider("Struct Th", 0.0, 0.5, 0.1); mode = st.radio("Mode", ["Fusion", "Structure Only", "Heatmap Only"], horizontal=True)
                    max_pts = st.select_slider("Points", [50000, 100000, 200000, 500000], value=100000)

            cam_x = st.slider("Camera X", -3.0, 3.0, 1.8); cam_y = st.slider("Camera Y", -3.0, 3.0, 1.8); cam_z = st.slider("Camera Z", -3.0, 3.0, 1.8)
            camera_eye = dict(x=cam_x, y=cam_y, z=cam_z)
            pts_args = dict(roi_bounds=(z_r, y_r, x_r), thresholds={'structure': th}, max_points=max_pts)

            st.markdown("### Channel 0: Lignin")
            pts0 = generate_point_cloud(vol_vis[..., 0], heatmap if mode != "Structure Only" else None, **pts_args)
            if mode == "Heatmap Only": pts0 = (pts0[0], pts0[1], pts0[2], pts0[3], pts0[4])
            fig0 = render_plotly_figure_v5(pts0, "", camera_eye, (D,H,W))
            st.plotly_chart(fig0, use_container_width=True)
            st.session_state.temp_pts0 = pts0

            if C > 1:
                st.markdown("### Channel 1: Cellulose")
                pts1 = generate_point_cloud(vol_vis[..., 1], heatmap if mode != "Structure Only" else None, **pts_args)
                fig1 = render_plotly_figure_v5(pts1, "", camera_eye, (D,H,W))
                st.plotly_chart(fig1, use_container_width=True)
                st.session_state.temp_pts1 = pts1

        with tab_exp:
            st.header("💾 Research Export Studio")
            base_name = f"{sample_name}_{arch}_Cls{st.session_state.pred_info[0]}"

            st.subheader("1. Save 3D Views")
            if st.button("📸 Save Current 3D"):
                if 'temp_pts0' in st.session_state:
                    f0 = render_plotly_figure_v5(st.session_state.temp_pts0, "", camera_eye, (D,H,W))
                    f0.write_image(os.path.join(DIRS["screenshots"], f"{base_name}_3D_Ch0.png"), scale=2)
                if 'temp_pts1' in st.session_state:
                    f1 = render_plotly_figure_v5(st.session_state.temp_pts1, "", camera_eye, (D,H,W))
                    f1.write_image(os.path.join(DIRS["screenshots"], f"{base_name}_3D_Ch1.png"), scale=2)
                st.success("Saved 3D Snapshots")

            st.subheader("2. Save 2D Slices (Red/Green Dual)")
            slice_idx = st.slider("Select Slice Z", 0, D-1, D//2)

            H_s, W_s, _ = vol_vis[slice_idx].shape
            rgb_preview = np.zeros((H_s, W_s, 3))
            rgb_preview[..., 0] = dynamic_rescaling(vol_vis[slice_idx, ..., 0])
            if C > 1: rgb_preview[..., 1] = dynamic_rescaling(vol_vis[slice_idx, ..., 1])

            fig_s, ax_s = plt.subplots(figsize=(5,5))
            ax_s.imshow(rgb_preview)
            hm_img = dynamic_rescaling(st.session_state.heatmap[slice_idx])
            ax_s.imshow(np.ma.masked_where(hm_img < 0.2, hm_img), cmap='jet', alpha=0.5)
            ax_s.axis('off')
            add_scale_bar_2d(ax_s, (H_s, W_s))
            st.pyplot(fig_s)

            if st.button("💾 Save Slice"):
                n_rg = f"{base_name}_L{slice_idx}_RG_Overlay.png"
                path_rg = save_rg_slice(vol_vis[slice_idx], st.session_state.heatmap[slice_idx], f"Slice {slice_idx}", n_rg)
                st.success(f"Saved: {path_rg}")

            st.subheader("3. Export Data")
            if st.button("📦 Export NPZ"):
                path = os.path.join(DIRS["npz"], f"{base_name}_Export.npz")
                np.savez_compressed(path, data=vol_vis, heatmap=heatmap)
                st.success(f"Saved: {path}")

if __name__ == "__main__":
    main()