import os
import glob
from collections import defaultdict
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from history.cell_fiber_dataset import CellFiberCleanDataset

def get_data_loaders(data_root, batch_size=12, num_workers=8):
    """
    自动读取 data_root 下的子文件夹作为分类类别
    修改：划分时确保同一原始样本的所有扩充数据（旋转/裁剪）都在同一个集中
    """
    if not os.path.exists(data_root):
        raise ValueError(f"Data root not found: {data_root}")

    # --- 1. 自动检测类别 ---
    CLASSES = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    CLASSES.sort()

    if len(CLASSES) < 2:
        print(f"⚠️  严重警告: 仅检测到 {len(CLASSES)} 个类别: {CLASSES}。请检查路径是否正确！")

    class_mapping = {cls_name: i for i, cls_name in enumerate(CLASSES)}
    print(f"✅ 检测到的类别 (自动): {CLASSES}")
    print(f"✅ 类别映射表: {class_mapping}")

    # --- 2. 获取所有文件路径 ---
    all_files = []
    for cls_name in CLASSES:
        cls_path = os.path.join(data_root, cls_name)
        files = glob.glob(os.path.join(cls_path, "*.npz"))
        all_files.extend(files)
        print(f"   - 类别 '{cls_name}': 发现 {len(files)} 个文件")

    if len(all_files) == 0:
        raise ValueError(f"在 {data_root} 下未找到任何 .npz 文件")

    # --- 3. 标签提取与划分 (核心修改) ---
    # 策略：先按照文件名特征将属于同一个样本的文件归类，然后按样本ID进行划分

    sample_groups = defaultdict(list)
    sample_labels_map = {}

    for f in all_files:
        basename = os.path.basename(f)
        # 提取样本ID:
        # 假设文件名为 E33_rot000_BL.npz，则通过分割 '_rot' 获取前缀 'E33' 作为唯一ID
        if '_rot' in basename:
            sample_id = basename.split('_rot')[0]
        else:
            # 兼容不包含 _rot 的情况，默认取第一个下划线前的内容
            sample_id = basename.split('_')[0]

        sample_groups[sample_id].append(f)

        # 获取该样本的类别 (父文件夹名)
        label = os.path.basename(os.path.dirname(f))
        sample_labels_map[sample_id] = label

    # 提取唯一的样本ID列表和对应的标签列表用于分层划分
    unique_ids = list(sample_groups.keys())
    unique_labels = [sample_labels_map[uid] for uid in unique_ids]

    print(f"🔍 识别到 {len(unique_ids)} 个独立原始样本 (扩充前)")

    # 划分: Train(80%) / Val(10%) / Test(10%) 基于 样本ID
    # 第一步：切分出 Test (10% 的样本ID)
    train_val_ids, test_ids, train_val_labels, test_labels = train_test_split(
        unique_ids, unique_labels, test_size=0.1, stratify=unique_labels, random_state=42
    )

    # 第二步：从剩余的样本ID中切分出 Val (总量的10%)
    train_ids, val_ids, train_labels, val_labels = train_test_split(
        train_val_ids, train_val_labels, test_size=1/9, stratify=train_val_labels, random_state=42
    )

    # 将划分好的 ID 列表还原为对应的所有文件路径
    def flatten_files(id_list):
        files = []
        for uid in id_list:
            files.extend(sample_groups[uid])
        return files

    train_files = flatten_files(train_ids)
    val_files = flatten_files(val_ids)
    test_files = flatten_files(test_ids)

    print(f"📊 数据集划分 (按样本ID) -> Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    print(f"📊 数据集划分 (总文件数) -> Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

    # --- 4. 创建 DataLoader ---
    train_dataset = CellFiberCleanDataset(train_files, class_mapping=class_mapping)
    val_dataset = CellFiberCleanDataset(val_files, class_mapping=class_mapping)
    test_dataset = CellFiberCleanDataset(test_files, class_mapping=class_mapping)

    # 针对 A800 开启 pin_memory 加速
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader