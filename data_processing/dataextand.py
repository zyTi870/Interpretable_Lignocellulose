import numpy as np
import os
import glob
import random
import time

# 尝试导入 cupy，如果没有安装则报错提示
try:
    import cupy as cp
    import cupyx.scipy.ndimage
    print("成功检测到 GPU环境 (CuPy)")
except ImportError:
    raise ImportError("请先安装 CuPy 以使用 GPU 加速。例如: pip install cupy-cuda11x")

def split_fixed_quadrants(volume):
    """
    固定切割 (CPU操作即可，切片很快，不需要GPU)
    """
    H, W = volume.shape[1], volume.shape[2]
    mid_h, mid_w = H // 2, W // 2

    quads = []
    quads.append(('fixed_TL', volume[:, 0:mid_h, 0:mid_w, :]))
    quads.append(('fixed_TR', volume[:, 0:mid_h, mid_w:W, :]))
    quads.append(('fixed_BL', volume[:, mid_h:H, 0:mid_w, :]))
    quads.append(('fixed_BR', volume[:, mid_h:H, mid_w:W, :]))

    return quads

def random_rotate_crop_gpu(volume_np, count=12, crop_size=256):
    """
    【GPU加速版】随机旋转 + 截取
    """
    generated = []

    # 1. 将数据从内存 (CPU) 搬运到 显存 (GPU)
    # volume_gpu 是位于显存中的数组
    volume_gpu = cp.asarray(volume_np)

    d, h, w, c = volume_gpu.shape

    margin = 80
    x_min, x_max = margin, h - margin - crop_size
    y_min, y_max = margin, w - margin - crop_size

    if x_min >= x_max: x_min, x_max = (h - crop_size)//2, (h - crop_size)//2 + 1
    if y_min >= y_max: y_min, y_max = (w - crop_size)//2, (w - crop_size)//2 + 1

    for i in range(count):
        angle = random.uniform(-180, 180)

        # 2. 在 GPU 上进行旋转
        # cupyx.scipy.ndimage.rotate 语法与 scipy 相同，但极快
        # order=1 (双线性), order=0 (最近邻，用于Mask)
        rot_vol_gpu = cupyx.scipy.ndimage.rotate(
            volume_gpu,
            angle,
            axes=(1, 2),
            reshape=False,
            mode='constant',
            cval=0,
            order=1
        )

        # 3. 在 GPU 上切片
        x = int(np.random.randint(x_min, x_max))
        y = int(np.random.randint(y_min, y_max))

        crop_gpu = rot_vol_gpu[:, x:x+crop_size, y:y+crop_size, :]

        # 4. 将结果从 显存 搬回 内存 (CPU) 以便保存
        crop_cpu = crop_gpu.get() # .get() 是 cupy 转 numpy 的方法

        suffix = f"rand_{i:02d}_ang{int(angle)}"
        generated.append((suffix, crop_cpu))

    # 显式释放显存 (可选，Python会自动回收，但循环量大时手动释放更安全)
    del volume_gpu
    del rot_vol_gpu
    cp.get_default_memory_pool().free_all_blocks()

    return generated

def process_dataset_gpu(input_dir, output_dir):
    files = glob.glob(os.path.join(input_dir, "**", "*.npz"), recursive=True)
    total_files = len(files)

    print(f"--- GPU 加速扩充脚本启动 ---")
    print(f"检测到文件: {total_files} 个")
    print(f"输入: {input_dir}")
    print(f"输出: {output_dir}")
    print("-" * 30)

    start_time = time.time()

    for idx, file_path in enumerate(files):
        try:
            # 1. 读取 (IO瓶颈，无法用GPU加速)
            data = np.load(file_path)
            if 'volume' not in data: continue
            volume = data['volume']

            if volume.shape[1] != 512 or volume.shape[2] != 512:
                continue

            # 2. 路径处理
            rel_path = os.path.relpath(file_path, input_dir)
            save_dir = os.path.join(output_dir, os.path.dirname(rel_path))
            os.makedirs(save_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(file_path))[0]

            save_count = 0

            # A. 固定切片 (CPU处理)
            fixed_patches = split_fixed_quadrants(volume)
            for suffix, patch_vol in fixed_patches:
                final_path = os.path.join(save_dir, f"{base_name}_{suffix}.npz")
                np.savez_compressed(final_path, volume=patch_vol)
                save_count += 1

            # B. 随机切片 (GPU处理核心计算)
            rand_patches = random_rotate_crop_gpu(volume, count=12, crop_size=256)
            for suffix, patch_vol in rand_patches:
                final_path = os.path.join(save_dir, f"{base_name}_{suffix}.npz")
                # 【重要提示】 save_compressed 非常慢，是主要限速步骤
                np.savez_compressed(final_path, volume=patch_vol)
                save_count += 1

            if (idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                speed = (idx + 1) / elapsed
                print(f"[{idx + 1}/{total_files}] 处理中... 速度: {speed:.2f} 张/秒")

        except Exception as e:
            print(f"错误: {e}")

    print(f"\n完成！总耗时: {time.time() - start_time:.2f} 秒")

if __name__ == "__main__":
    INPUT_ROOT = "/home/xxge/tzy/Pycharmpro/cnn_fiber/TRAIN_DATA_NPZ"
    OUTPUT_ROOT = "/home/xxge/tzy/Pycharmpro/cnn_fiber/TRAIN_DATA_FINAL_256"

    if os.path.exists(INPUT_ROOT):
        process_dataset_gpu(INPUT_ROOT, OUTPUT_ROOT)
    else:
        print("路径不存在")