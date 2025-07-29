from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def activation_factory(name: str = "lrelu") -> nn.Module:
    return nn.ReLU() if name == "relu" else nn.LeakyReLU(negative_slope=0.2) if name == "lrelu" else nn.Identity()


class ConvBlock3d(nn.Module):
    def __init__(self, c_in, c_out, use_bn: bool, act: str):
        super().__init__()
        self.conv = nn.Conv3d(c_in, c_out, 3, 1, 1)
        self.bn = nn.BatchNorm3d(c_out) if use_bn else nn.Identity()
        self.act = activation_factory(act)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)

        # print(f"ConvBlock3d output shape: {x.shape}")  # Debugging output

        return x


class Encoder3D(nn.Module):
    def __init__(self, in_ch: int, base_ch: int, layers: int, use_bn: bool, act: str):
        super().__init__()
        blocks = []
        c_in, c = in_ch, base_ch
        for _ in range(layers):
            blocks.append(ConvBlock3d(c_in, c, use_bn, act))
            c_in, c = c, c * 2
        self.blocks = nn.Sequential(*blocks)
        self.out_ch = c_in  # channels after last block

    def forward(self, x):
        return self.blocks(x)  # (B, C, D', H', W')


class MLPHead(nn.Module):
    """Shared MLP head template -> outputs (B, K, 4)."""
    def __init__(self, in_feat: int, k: int, act: str):
        super().__init__()
        hidden = max(64, in_feat // 2)
        self.k = k
        self.mlp = nn.Sequential(
            nn.Linear(in_feat, hidden), activation_factory(act),
            nn.Linear(hidden, hidden // 2), activation_factory(act),
            nn.Linear(hidden // 2, k * 4),
        )

    def forward(self, f_flat):
        B = f_flat.size(0)
        out = self.mlp(f_flat)           # (B, K*4)
        out = out.view(B, self.k, 4)     # (B, K, 4)
        return out


class PRSNet(nn.Module):
    """
    Minimal, clean architecture:
      voxel (B,1,gs,gs,gs) -> encoder -> GAP -> planes(Kp,4), quats(Kq,4)
    """
    def __init__(self,
                 input_nc: int = 1,
                 base_channels: int = 32,
                 conv_layers: int = 5,
                 num_planes: int = 3,
                 num_quats: int = 3,
                 use_bn: bool = False,
                 activation: str = "lrelu",
                 dropout: float = 0.0):
        super().__init__()
        self.encoder = Encoder3D(input_nc, base_channels, conv_layers, use_bn, activation)
        self.gap = nn.AdaptiveAvgPool3d(1)
        feat_dim = self.encoder.out_ch
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        self.plane_head = MLPHead(feat_dim, num_planes, activation)
        self.quat_head  = MLPHead(feat_dim, num_quats, activation)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    @staticmethod
    def _normalize_planes(p: torch.Tensor) -> torch.Tensor:
        # normalize first 3 dims to unit length; keep d unconstrained
        n = p[..., :3]
        d = p[..., 3:4]
        n = n / (n.norm(p=2, dim=-1, keepdim=True) + 1e-12)
        return torch.cat([n, d], dim=-1)

    @staticmethod
    def _normalize_quats(q: torch.Tensor) -> torch.Tensor:
        # normalize full quaternion to unit norm
        return q / (q.norm(p=2, dim=-1, keepdim=True) + 1e-12)

    def forward(self, voxel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            quats:  (B, Kq, 4)  unit quaternions
            planes: (B, Kp, 4)  unit normals, free d
        """
        feat = self.encoder(voxel)              # (B,C,D,H,W)
        feat = self.gap(feat).flatten(1)        # (B,C)
        feat = self.dropout(feat)

        planes = self._normalize_planes(self.plane_head(feat))
        quats  = self._normalize_quats(self.quat_head(feat))
        return quats, planes

if __name__ == "__main__":
    # Example usage
    model = PRSNet(input_nc=1, base_channels=4, conv_layers=5, num_planes=3, num_quats=3, use_bn=True, activation='lrelu')
    voxel_input = torch.randn(1, 1, 32, 32, 32)  # Batch of 1 voxel
    quats, planes = model(voxel_input)
    print("Quaternions shape:", quats.shape)  # Should be (1, 3, 4)
    print("Planes shape:", planes.shape)      # Should be (1, 3, 4)
