import torch
import torch.nn as nn

class PatchEmbedding3D(nn.Module):
    def __init__(self, img_size=(64, 256, 256), patch_size=(8, 16, 16), in_channels=2, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size

        # 计算 Patch 数量
        self.num_patches = (img_size[0] // patch_size[0]) * \
                           (img_size[1] // patch_size[1]) * \
                           (img_size[2] // patch_size[2])

        # 使用 Conv3d 实现 Patch 投影
        # Kernel size = stride = patch_size 实现了无重叠切块
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, D, H, W) -> (B, Embed_Dim, D_p, H_p, W_p)
        x = self.proj(x)
        # Flatten: (B, Embed_Dim, Num_Patches)
        x = x.flatten(2)
        # Transpose: (B, Num_Patches, Embed_Dim)
        x = x.transpose(1, 2)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
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
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ViT3D(nn.Module):
    # 修改点：embed_dim 从 768 降为 384，depth 从 12 降为 8，num_heads 从 12 降为 6
    def __init__(self, img_size=(64, 256, 256), patch_size=(8, 16, 16), in_channels=2, num_classes=5,
                 embed_dim=384, depth=8, num_heads=6, mlp_ratio=4., qkv_bias=True, drop_rate=0.):
        super().__init__()

        self.patch_embed = PatchEmbedding3D(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.ModuleList([
            TransformerBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Init weights
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
        # x: (B, C, D, H, W)
        B = x.shape[0]
        x = self.patch_embed(x) # (B, N, E)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        # 取 cls_token 的输出
        return self.head(x[:, 0])

# --- 验证代码 (你可以运行下面这段代码来确认参数量) ---
if __name__ == '__main__':
    model = ViT3D()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params / 1e6:.2f}M")
    # 输出结果约为: 16.52M