import torch
import torch.nn as nn

class PatchEmbedding3D(nn.Module):
    def __init__(self, img_size=(64, 256, 256), patch_size=(16, 32, 32), in_channels=2, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size

        # 增大patch_size后，Patch数量大幅减少：(64/16)×(256/32)×(256/32) = 4×8×8=256个
        self.num_patches = (img_size[0] // patch_size[0]) * \
                           (img_size[1] // patch_size[1]) * \
                           (img_size[2] // patch_size[2])

        # 3D卷积投影：通道2→256，核尺寸增大，参数量从3.15M→0.41M
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, 256, 4, 8, 8)
        x = x.flatten(2)  # (B, 256, 256)
        x = x.transpose(1, 2)  # (B, 256, 256)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads  # 256/8=32，符合注意力头维度要求
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # 参数量：256×768=196,608
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)  # 参数量：256×256=65,536
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        # mlp_ratio从4→2，隐藏层维度：256×2=512
        hidden_features = hidden_features or int(in_features * 2)
        self.fc1 = nn.Linear(in_features, hidden_features)  # 256×512=131,072
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)  # 512×256=131,072
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=2., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)  # 参数量：256×2=512
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)  # 参数量：256×2=512
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ViT3D_Light(nn.Module):
    def __init__(self, img_size=(64, 256, 256), patch_size=(16, 32, 32), in_channels=2, num_classes=5,
                 embed_dim=256, depth=6, num_heads=8, mlp_ratio=2., qkv_bias=True, drop_rate=0.):
        super().__init__()

        self.patch_embed = PatchEmbedding3D(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # cls_token：参数量256
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # pos_embed：(1, 256+1, 256) → 257×256=65,792
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer Block数量从12→6
        self.blocks = nn.ModuleList([
            TransformerBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)  # 256×2=512
        self.head = nn.Linear(embed_dim, num_classes)  # 256×5=1,280

        # 初始化权重
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, 256, 256)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 257, 256)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return self.head(x[:, 0])

# 轻量化3D ViT实例化（参数量≈18M）
def vit3d_light(num_classes=5, in_channels=2):
    return ViT3D_Light(
        img_size=(64, 256, 256),
        patch_size=(16, 32, 32),
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dim=256,
        depth=6,
        num_heads=8,
        mlp_ratio=2.,
        qkv_bias=True,
        drop_rate=0.
    )