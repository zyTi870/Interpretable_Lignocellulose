import os
import torch
import numpy as np
from torch.utils.data import Dataset

class CellFiberCleanDataset(Dataset):
    def __init__(self, file_paths, target_depth=64, class_mapping=None):
        """
        Args:
            file_paths (list): npz 文件路径列表
            target_depth (int): 目标深度 D_target = 64
            class_mapping (dict): 类别名称到整数的映射，例如 {'A': 0, 'E': 1...}
        """
        self.file_paths = file_paths
        self.target_depth = target_depth
        self.class_mapping = class_mapping

    def __len__(self):
        return len(self.file_paths)

    def _normalize(self, volume):
        """
        数据规范化逻辑：归一化到 0.0 - 1.0
        """
        # 转换为 float32 以避免计算溢出
        volume = volume.astype(np.float32)

        # 动态获取最大最小值，兼容 0-255 或 0-65535 的数据
        min_val = volume.min()
        max_val = volume.max()

        # 避免除以零
        if max_val - min_val > 0:
            volume = (volume - min_val) / (max_val - min_val)
        else:
            volume = volume - min_val # 全黑或全白图像

        return volume

    def _process_depth(self, volume):
        """
        深度处理：去除空白层逻辑需结合实际数据，这里主要实现
        补零/截取到 D=64 的核心逻辑
        """
        current_depth = volume.shape[0] # (D, H, W, C)

        if current_depth == self.target_depth:
            return volume

        if current_depth > self.target_depth:
            # 截取：取中间部分（假设中间部分信息量最大）
            start = (current_depth - self.target_depth) // 2
            end = start + self.target_depth
            volume = volume[start:end, :, :, :]
        else:
            # 补零：在深度方向两端补零
            pad_total = self.target_depth - current_depth
            pad_before = pad_total // 2
            pad_after = pad_total - pad_before
            # np.pad 格式: ((before_D, after_D), (H, H), (W, W), (C, C))
            volume = np.pad(volume,
                            ((pad_before, pad_after), (0, 0), (0, 0), (0, 0)),
                            mode='constant', constant_values=0)
        return volume

    def __getitem__(self, idx):
        path = self.file_paths[idx]

        # --- 1. 标签加载 ---
        # 路径示例: .../TRAIN_DATA_FINAL_256/E/E99_rot270_TR.npz
        # 获取父文件夹名称作为标签
        parent_dir = os.path.basename(os.path.dirname(path))

        # 如果标签在文件名中也有体现，也可以解析文件名。这里优先使用文件夹结构。
        label_str = parent_dir

        if self.class_mapping is None or label_str not in self.class_mapping:
            # 容错处理，或者抛出异常
            raise ValueError(f"Unknown label '{label_str}' in path: {path}")

        label = self.class_mapping[label_str]

        # --- 加载数据 ---
        # 假设 npz 中主要数据的 key 为 'arr_0' 或 'data'，需根据实际情况调整
        try:
            with np.load(path) as data:
                # 尝试常见的 key，或者直接获取第一个数组
                if 'data' in data:
                    volume = data['data']
                elif 'arr_0' in data:
                    volume = data['arr_0']
                else:
                    volume = data[list(data.keys())[0]]
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # 返回全0数据防止崩溃，实际工程中应记录日志
            return torch.zeros((2, self.target_depth, 256, 256)), label

        # 原始形状: (D, H, W, C) -> (D, 256, 256, 2)

        # --- 深度处理 (截取/补零) ---
        volume = self._process_depth(volume)

        # --- 2. 数据规范化 (0.0 - 1.0) ---
        volume = self._normalize(volume)

        # --- 格式转换 ---
        # Numpy -> Tensor
        volume_tensor = torch.from_numpy(volume)

        # 调整维度顺序以符合 PyTorch 3D Conv: (N, C, D, H, W)
        # 当前是 (D, H, W, C) -> 变为 (C, D, H, W)
        volume_tensor = volume_tensor.permute(3, 0, 1, 2)

        # 确保数据类型为 float32 (PyTorch 默认)
        volume_tensor = volume_tensor.float()

        return volume_tensor, torch.tensor(label, dtype=torch.long)