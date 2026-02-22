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

# --- 基础配置 ---
st.set_page_config(layout="wide", page_title="CNN Fiber 3D XAI v3.6 (Vertical 3D + Black Bar)")
sys.path.append(os.getcwd())

# 设置科研级字体
try:
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'stix'
except:
    pass

# 根文件夹名称
ROOT_OUTPUT_DIR = "../3d_processing_outputs"

# 子目录配置
DIRS = {
    "screenshots": os.path.join(ROOT_OUTPUT_DIR, "output_3d_snapshots"),
    "slices": os.path.join(ROOT_OUTPUT_DIR, "output_2d_slices"),
    "npz": os.path.join(ROOT_OUTPUT_DIR, "output_processed_npz")
}
for d in DIRS.values():
    os.makedirs(d, exist_ok=True)

# ==========================================
# 0. 核心工具函数
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

def add_scale_bar_2d(ax, image_width_px, pixel_size_um=1.32, bar_length_um=50, location='lower right'):
    """Matplotlib 2D 比例尺"""
    bar_pixels = bar_length_um / pixel_size_um
    H, W = image_width_px
    pad = 10
    if location == 'lower right':
        x0 = W - bar_pixels - pad - 10
        y0 = H - pad - 5
    rect = Rectangle((x0, y0), bar_pixels, H*0.015, color='white', linewidth=0)
    ax.add_patch(rect)
    font_args = {'color': 'white', 'ha': 'center', 'va': 'bottom', 'fontsize': 10, 'fontweight': 'bold'}
    try:
        ax.text(x0 + bar_pixels/2, y0 - 10, f'{bar_length_um} $\mu m$', fontname='Times New Roman', **font_args)
    except:
        ax.text(x0 + bar_pixels/2, y0 - 10, f'{bar_length_um} um', **font_args)

# ==========================================
# 1. 模型加载器
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

# ==========================================
# 2. XAI 引擎
# ==========================================
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
            b, n, e = g.shape
            d, h, w = 8, 16, 16
            g = g.reshape(b, d, h, w, e).permute(0, 4, 1, 2, 3)
            a = a.reshape(b, d, h, w, e).permute(0, 4, 1, 2, 3)
        weights = torch.mean(g, dim=(2,3,4), keepdim=True)
        cam = torch.sum(weights * a, dim=1, keepdim=True)
        cam = F.relu(cam)
        return self._resize_and_norm(cam, x.shape[2:])
    def run_cbam(self, model, x):
        self.clear()
        target = None
        for m in model.modules():
            if isinstance(m, SpatialAttention3D): target = m
        if not target: return None, "No CBAM module found"
        self.hooks.append(target.register_forward_hook(lambda m,i,o: setattr(self, 'data', o.detach())))
        with torch.no_grad(): model(x)
        return self._resize_and_norm(self.data, x.shape[2:]), "OK"
    def run_vit_attn(self, model, x):
        self.clear()
        target = model.blocks[-1].attn.attn_drop if hasattr(model, 'blocks') else None
        if not target: return None, "No ViT Attention found"
        self.hooks.append(target.register_forward_hook(lambda m,i,o: setattr(self, 'data', i[0].detach())))
        with torch.no_grad(): model(x)
        attn = torch.mean(self.data, dim=1)[:, 0, 1:]
        b, n = attn.shape
        attn = attn.reshape(b, 1, 8, 16, 16)
        return self._resize_and_norm(attn, x.shape[2:]), "OK"

# ==========================================
# 3. 可视化引擎
# ==========================================
def generate_safe_point_cloud(volume, heatmap, roi_bounds, thresholds, max_points=150000):
    D, H, W = volume.shape
    z, y, x = np.indices((D, H, W))
    hm_norm = dynamic_rescaling(heatmap)
    vol_norm = dynamic_rescaling(volume)
    (z_min, z_max), (y_min, y_max), (x_min, x_max) = roi_bounds
    roi_mask = ((z >= z_min) & (z < z_max) & (y >= y_min) & (y < y_max) & (x >= x_min) & (x < x_max))
    struct_mask = vol_norm > thresholds['structure']
    final_mask = roi_mask & struct_mask
    pts_x = x[final_mask]; pts_y = y[final_mask]; pts_z = z[final_mask]
    pts_v = vol_norm[final_mask]; pts_h = hm_norm[final_mask]
    num_points = len(pts_x)
    if num_points > max_points:
        high_heat_indices = np.where(pts_h > 0.4)[0]
        low_heat_indices = np.where(pts_h <= 0.4)[0]
        quota = max_points - len(high_heat_indices)
        if quota > 0 and len(low_heat_indices) > 0:
            sampled_low = np.random.choice(low_heat_indices, quota, replace=False)
            keep_indices = np.concatenate([high_heat_indices, sampled_low])
        else:
            if len(high_heat_indices) > max_points:
                keep_indices = np.random.choice(high_heat_indices, max_points, replace=False)
            else:
                keep_indices = high_heat_indices
        pts_x = pts_x[keep_indices]; pts_y = pts_y[keep_indices]; pts_z = pts_z[keep_indices]
        pts_v = pts_v[keep_indices]; pts_h = pts_h[keep_indices]
    return pts_x, pts_y, pts_z, pts_v, pts_h

def render_plotly_figure_v3_6(pts_data, title, fusion_mode, camera_eye, volume_shape):
    """V3.6: 黑色细标尺 + 上下显示 + 坐标轴回归"""
    px, py, pz, pv, ph = pts_data
    if fusion_mode == 'Structure Only': color_val = pv; colorscale = 'Bone'; opacity = 0.1
    elif fusion_mode == 'Heatmap Only': color_val = ph; colorscale = 'Jet'; opacity = 0.1
    else: color_val = ph; colorscale = 'Jet'; opacity = 0.25

    # 1. 散点数据
    data = [go.Scatter3d(
        x=px, y=py, z=pz, mode='markers',
        marker=dict(size=2, color=color_val, colorscale=colorscale, opacity=opacity, colorbar=dict(title="Intensity")),
        hovertext=[f"V:{v:.2f}, H:{h:.2f}" for v, h in zip(pv, ph)],
        name=title
    )]

    # 2. 标尺 (Scale Bar) - 改为黑色细线
    D, H, W = volume_shape
    scale_um = 50
    pixel_size = 1.32
    bar_len_px = scale_um / pixel_size
    offset = 10

    x_bar = [W - offset - bar_len_px, W - offset]
    y_bar = [H - offset, H - offset]
    z_bar = [0, 0]

    data.append(go.Scatter3d(
        x=x_bar, y=y_bar, z=z_bar,
        mode='lines+text',
        # 【修改】黑色 + 细线 (width=3)
        line=dict(color='black', width=3),
        text=[f"{scale_um} um", ""],
        textposition="top center",
        # 【修改】黑色字体
        textfont=dict(color='black', size=14, family='Times New Roman'),
        showlegend=False,
        name="Scale Bar"
    ))

    scene_camera = dict(eye=camera_eye) if camera_eye else None

    # 3. 坐标轴配置 (回归)
    # 【修改】Z轴还是要标，且开启网格
    axis_config = dict(
        showgrid=True,        # 开启网格
        gridcolor='rgba(200, 200, 200, 0.5)', # 淡灰色网格
        showbackground=False,
        showline=True,        # 开启轴线
        linecolor='black',
        linewidth=2,
        showticklabels=True,  # 开启数值
        tickfont=dict(size=10, color='black'),
        title_font=dict(size=12, color='black'),
        zeroline=False,
        visible=True
    )

    layout = go.Layout(
        title=title, autosize=True, height=600,
        scene=dict(
            xaxis=dict(title='Width (X)', **axis_config),
            yaxis=dict(title='Height (Y)', **axis_config),
            zaxis=dict(title='Depth (Z)', **axis_config),
            aspectmode='data', bgcolor="rgba(0,0,0,0)",
            camera=scene_camera, dragmode='turntable'
        ),
        margin=dict(l=50, r=50, b=50, t=50),
        paper_bgcolor="rgba(255, 255, 255, 0)"
    )
    return go.Figure(data=data, layout=layout)

def save_research_slice(vol_slice, hm_slice, title_str, save_name):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    v_img = dynamic_rescaling(vol_slice)
    h_img = dynamic_rescaling(hm_slice)
    ax.imshow(v_img, cmap='bone', alpha=1.0)
    h_masked = np.ma.masked_where(h_img < 0.2, h_img)
    ax.imshow(h_masked, cmap='jet', alpha=0.5)
    ax.axis('off')
    try: ax.set_title(title_str, fontname='Arial', fontsize=14, pad=10)
    except: ax.set_title(title_str, fontsize=14, pad=10)
    add_scale_bar_2d(ax, v_img.shape, pixel_size_um=1.32, bar_length_um=50)
    path = os.path.join(DIRS["slices"], save_name)
    plt.savefig(path, bbox_inches='tight', pad_inches=0.05, dpi=300)
    plt.close(fig)
    return path

# ==========================================
# 4. 主程序
# ==========================================
def main():
    def main():
        st.markdown("## 🔬 CNN Fiber 3D XAI v3.6 (Vertical 3D + Black Bar)")

    with st.sidebar:
        st.header("1. Data & Model Setup")
        arch = st.selectbox("Architecture", ["ResNet50", "ResNet18", "DenseNet121", "ViT"])
        use_cbam = st.checkbox("Enable CBAM", True)

        # --- 模型选择：仅显示文件名 ---
        ckpt_dir = "./checkpoints_cbam" if os.path.exists("../checkpoints_cbam") else "."
        ckpts = [os.path.join(r, f) for r, d, fl in os.walk(ckpt_dir) for f in fl if f.endswith(".pth")]
        if not ckpts:
            st.error("No Checkpoints found"); st.stop()

        ckpt = st.selectbox(
            "Checkpoint",
            options=ckpts,
            format_func=lambda x: os.path.basename(x)  # 只显示文件名
        )

        # --- 数据文件选择：仅显示文件名 ---
        data_dir = "./TRAIN_DATA_FINAL_256" if os.path.exists("../TRAIN_DATA_FINAL_256") else "."
        datas = [os.path.join(r, f) for r, d, fl in os.walk(data_dir) for f in fl if f.endswith(".npz")]
        if not datas:
            st.error("No Data found"); st.stop()

        f_path = st.selectbox(
            "Input Sample (.npz)",
            options=datas,
            format_func=lambda x: os.path.basename(x)  # 只显示文件名
        )

        model, msg = load_model_resource(arch, 5, use_cbam, ckpt)
        if not model: st.error(msg); st.stop()
        else: st.success(f"Model Loaded: {arch}")
        st.divider()
        st.header("2. Linked Camera Control")
        st.info("💡 提示：黑色标尺(3px)。坐标轴已恢复。")
        cam_x = st.slider("Camera Eye X", -3.0, 3.0, 2.0, 0.1)
        cam_y = st.slider("Camera Eye Y", -3.0, 3.0, 2.0, 0.1)
        cam_z = st.slider("Camera Eye Z", -3.0, 3.0, 2.0, 0.1)
        camera_eye = dict(x=cam_x, y=cam_y, z=cam_z)

    # 数据加载
    try:
        with np.load(f_path) as f: raw = f['data'] if 'data' in f else f[list(f.keys())[0]]
        if raw.ndim == 5: raw = raw[0]
        tensor_in, vol_vis = preprocess_volume(raw)
    except Exception as e: st.error(f"Data Load Error: {e}"); st.stop()

    engine = UltimateXAIEngine()
    x = torch.from_numpy(tensor_in).unsqueeze(0).float().to(engine.device)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        pred_class = probs.argmax().item()
        pred_conf = probs[0][pred_class].item()

    if 'heatmap' not in st.session_state or st.session_state.get('last_file') != f_path:
        st.session_state.heatmap = None
        st.session_state.last_file = f_path

    # --- Tab 逻辑 ---
    tab_calc, tab_vis, tab_export = st.tabs(["📊 Analysis Engine", "🖼️ 3D Visualization", "💾 Research Export Studio"])

    with tab_calc:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Prediction", f"Class {pred_class}", f"{pred_conf*100:.2f}%")
            method = st.radio("XAI Method", ["Grad-CAM", "CBAM Attention", "ViT Attention"])
        with c2:
            layer_name = None
            if method == "Grad-CAM":
                if "ResNet" in arch:
                    l_sel = st.selectbox("Layer", ["layer4", "layer3", "layer2"], index=1)
                    layer_name = l_sel
                elif "DenseNet" in arch:
                    l_sel = st.selectbox("Block", ["denseblock4", "denseblock3"], index=0)
                    layer_name = l_sel

            if st.button("🚀 Calculate Attention Map", type="primary"):
                with st.spinner("Analyzing..."):
                    hm = None
                    if method == "Grad-CAM":
                        target_layer = None
                        if "ResNet" in arch:
                            block = getattr(model, layer_name)
                            target_layer = block[-1].conv3 if hasattr(block[-1], 'conv3') else block[-1].conv2
                        elif "DenseNet" in arch:
                            for n, m in model.features.named_children():
                                if n == layer_name: target_layer = m
                        if target_layer: hm = engine.run_gradcam(model, x, target_layer, pred_class)
                    elif method == "CBAM Attention": hm, _ = engine.run_cbam(model, x)
                    elif method == "ViT Attention": hm, _ = engine.run_vit_attn(model, x)

                    if hm is None: st.warning("Analysis failed.")
                    else: st.session_state.heatmap = hm; st.success("Heatmap Generated!")

    # --- 核心修改：3D Visualization 标签页 ---
    with tab_vis:
        if st.session_state.heatmap is None:
            st.info("Please calculate attention map first.")
        else:
            heatmap = st.session_state.heatmap
            D, H, W, C = vol_vis.shape

            with st.expander("Rendering Controls", expanded=True):
                c_roi, c_style = st.columns(2)
                with c_roi:
                    z_r = st.slider("Z Clip", 0, D, (0, D)); y_r = st.slider("Y Clip", 0, H, (0, H)); x_r = st.slider("X Clip", 0, W, (0, W))
                with c_style:
                    th = st.slider("Struct Th", 0.0, 0.5, 0.1); mode = st.radio("Mode", ["Fusion", "Structure", "Heatmap"], horizontal=True)
                    max_pts = st.select_slider("Points", options=[50000, 100000, 200000], value=100000)

            col_ch0, col_ch1 = st.columns(2)
            pts_args = dict(roi_bounds=(z_r, y_r, x_r), thresholds={'structure': th}, max_points=max_pts)

            # --- 垂直显示 (不再使用 col_ch0, col_ch1 左右布局) ---
            # 1. Channel 0
            st.markdown("### Channel 0: Lignin")
            if C > 0:
                pts0 = generate_safe_point_cloud(vol_vis[..., 0], heatmap, **pts_args)
                fig0 = render_plotly_figure_v3_6(pts0, "", mode, camera_eye, (D,H,W))
                st.plotly_chart(fig0, use_container_width=True, key="p0")

            # 2. Channel 1
            st.markdown("### Channel 1: Cellulose")
            if C > 1:
                pts1 = generate_safe_point_cloud(vol_vis[..., 1], heatmap, **pts_args)
                fig1 = render_plotly_figure_v3_6(pts1, "", mode, camera_eye, (D,H,W))
                st.plotly_chart(fig1, use_container_width=True, key="p1")

            if 'pts0' in locals(): st.session_state.temp_pts0 = pts0
            if 'pts1' in locals(): st.session_state.temp_pts1 = pts1

    # Tab 3
    with tab_export:
        if st.session_state.heatmap is None:
            st.warning("Data not ready.")
        else:
            st.markdown("### 💾 Research Export Studio")
            heatmap = st.session_state.heatmap
            D, H, W, _ = vol_vis.shape
            base_name = f"{arch}_{method}_Class{pred_class}_{datetime.now().strftime('%H%M%S')}"

            # 1. 3D 截图
            st.subheader("1. Save Current 3D View")
            if st.button("📸 Save 3D Views"):
                try:
                    if 'temp_pts0' in st.session_state:
                        # 导出时也用 V3.6 的渲染风格
                        fig0 = render_plotly_figure_v3_6(st.session_state.temp_pts0, "", mode, camera_eye, (D,H,W))
                        fig1 = render_plotly_figure_v3_6(st.session_state.temp_pts1, "", mode, camera_eye, (D,H,W))
                        p0 = os.path.join(DIRS["screenshots"], f"{base_name}_Ch0.png")
                        p1 = os.path.join(DIRS["screenshots"], f"{base_name}_Ch1.png")
                        fig0.write_image(p0, scale=2); fig1.write_image(p1, scale=2)
                        st.success(f"Saved: {p0}")
                except Exception as e: st.error(f"Save failed: {e}")

            st.divider()

            # 2. 2D 切片
            st.subheader("2. Save 2D Slices (Research Grade)")
            c_ctrl1, c_ctrl2 = st.columns([2, 1])
            with c_ctrl1:
                slice_idx = st.slider("Select Z-Layer Index", 0, D-1, D//2)
            with c_ctrl2:
                if st.button("💾 Render & Save Slices"):
                    n0 = f"{base_name}_L{slice_idx}_Ch0.png"
                    n1 = f"{base_name}_L{slice_idx}_Ch1.png"
                    path0 = save_research_slice(vol_vis[slice_idx,:,:,0], heatmap[slice_idx], f"{arch}|Ch0", n0)
                    path1 = save_research_slice(vol_vis[slice_idx,:,:,1], heatmap[slice_idx], f"{arch}|Ch1", n1)
                    st.success(f"Saved Slices to: {path0}")

            st.markdown("#### Preview: Channel 0 (Lignin)")
            fig_s0, ax_s0 = plt.subplots(figsize=(6, 6))
            v0 = dynamic_rescaling(vol_vis[slice_idx,:,:,0])
            hm0 = dynamic_rescaling(heatmap[slice_idx])
            ax_s0.imshow(v0, cmap='bone')
            ax_s0.imshow(np.ma.masked_where(hm0 < 0.2, hm0), cmap='jet', alpha=0.5)
            ax_s0.axis('off')
            add_scale_bar_2d(ax_s0, v0.shape)
            st.pyplot(fig_s0, use_container_width=False)

            st.markdown("#### Preview: Channel 1 (Cellulose)")
            fig_s1, ax_s1 = plt.subplots(figsize=(6, 6))
            v1 = dynamic_rescaling(vol_vis[slice_idx,:,:,1])
            ax_s1.imshow(v1, cmap='bone')
            ax_s1.imshow(np.ma.masked_where(hm0 < 0.2, hm0), cmap='jet', alpha=0.5)
            ax_s1.axis('off')
            add_scale_bar_2d(ax_s1, v1.shape)
            st.pyplot(fig_s1, use_container_width=False)

            st.divider()

            # 3. NPZ
            st.subheader("3. Save Processed NPZ")
            if st.button("📦 Export NPZ"):
                export_data = np.stack([vol_vis[...,0], vol_vis[...,1], heatmap], axis=0)
                path = os.path.join(DIRS["npz"], f"{base_name}_Processed.npz")
                np.savez_compressed(path, data=export_data)
                st.success(f"Saved: {path}")

if __name__ == "__main__":
    main()