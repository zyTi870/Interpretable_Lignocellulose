# 修改 __init__.py 文件内容

from .resnet3d import resnet50_3d, resnet18_3d
from .densenet3d import densenet121_3d
from .vit3d import ViT3D

def get_model(model_name, num_classes=5, in_channels=2):
    if model_name == 'resnet50':
        return resnet50_3d(num_classes=num_classes, in_channels=in_channels, use_cbam=False)
    elif model_name == 'resnet50_cbam':
        return resnet50_3d(num_classes=num_classes, in_channels=in_channels, use_cbam=True)

    elif model_name == 'resnet18':
        return resnet18_3d(num_classes=num_classes, in_channels=in_channels, use_cbam=False)
    elif model_name == 'resnet18_cbam':
        return resnet18_3d(num_classes=num_classes, in_channels=in_channels, use_cbam=True)

    # --- DenseNet 部分 ---
    elif model_name == 'densenet121':
        return densenet121_3d(num_classes=num_classes, in_channels=in_channels, use_cbam=False)
    elif model_name == 'densenet121_cbam': # 新增调用
        return densenet121_3d(num_classes=num_classes, in_channels=in_channels, use_cbam=True)
    # --------------------

    elif model_name == 'vit':
        return ViT3D(num_classes=num_classes, in_channels=in_channels)
    else:
        raise ValueError(f"Unknown model name: {model_name}")