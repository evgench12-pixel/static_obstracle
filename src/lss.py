"""Lift-Splat-Shoot adapted to our 4-camera BEV occupancy task.

Reference: https://arxiv.org/abs/2008.05711

Pipeline per forward pass:
  1. Each camera image runs through a shared backbone → feature map (C', H', W').
  2. A 1x1 head predicts D depth-bin softmax weights and a C-dim feature per
     feature pixel; their outer product builds a frustum-shaped volume of
     features per camera (D, C, H', W').
  3. We unproject feature pixels at each depth bin into 3D points using the
     camera intrinsics, then transform to the ego (car) frame using car2cam.
  4. We sum-pool features whose 3D points fall into each BEV cell — this is
     the "splat" step.
  5. A small CNN on the BEV grid produces the final occupancy logits.

The model takes (images, intrinsics, car2cams) and returns logits (B, 1, 188, 126).
Car frame convention used here: X=forward, Y=right, Z=up (per contest spec).
"""
import torch
import torch.nn as nn
import torchvision

from src.config import IMAGE_SIZE, OUT_SIZE


# BEV grid spec (matches the GT shape and the visualization extent in baseline)
BEV_X_RANGE = (0.0, 150.4)    # forward: 188 cells × 0.8m
BEV_Y_RANGE = (-50.4, 50.4)   # lateral: 126 cells × 0.8m
BEV_RES = 0.8

# Depth-bin range for the lift step
DEPTH_RANGE = (2.0, 80.0)
DEPTH_BINS = 39   # 2.0 step
DOWNSAMPLE = 32   # ResNet18 final stride


def make_frustum(image_size, downsample, depth_range, depth_bins):
    """Per-camera frustum points (D, H', W', 3) in image+depth coords (u*d, v*d, d)."""
    H, W = image_size
    Hf, Wf = H // downsample, W // downsample

    d_min, d_max = depth_range
    d_step = (d_max - d_min) / depth_bins
    depths = torch.linspace(d_min + d_step / 2, d_max - d_step / 2, depth_bins)

    u = torch.linspace(0, W - 1, Wf)
    v = torch.linspace(0, H - 1, Hf)

    d_grid = depths[:, None, None].expand(-1, Hf, Wf)
    u_grid = u[None, None, :].expand(depth_bins, Hf, -1)
    v_grid = v[None, :, None].expand(depth_bins, -1, Wf)

    return torch.stack([u_grid * d_grid, v_grid * d_grid, d_grid], dim=-1)  # (D, Hf, Wf, 3)


def get_geometry(frustum, intrinsics, car2cams):
    """Transform per-camera frustum points to the ego (car) frame.

    Computed in fp32 regardless of autocast — torch.linalg.inv is unstable in fp16.

    frustum: (D, H', W', 3) in image+depth coords
    intrinsics: (B, N, 3, 3) camera intrinsic K
    car2cams: (B, N, 4, 4) — semantically camera→car (despite the contest naming)

    Returns: (B, N, D, H', W', 3) — points in car frame, fp32.
    """
    intrinsics = intrinsics.float()
    car2cams = car2cams.float()
    frustum = frustum.float()

    B, N = intrinsics.shape[:2]
    D, H, W, _ = frustum.shape

    ones = torch.ones(D, H, W, 1, device=frustum.device, dtype=frustum.dtype)
    frust_h = torch.cat([frustum, ones], dim=-1)  # (D, H, W, 4)

    inv_K = torch.zeros(B, N, 4, 4, device=intrinsics.device, dtype=intrinsics.dtype)
    inv_K[..., :3, :3] = torch.linalg.inv(intrinsics)
    inv_K[..., 3, 3] = 1.0

    transform = (car2cams @ inv_K).view(B, N, 1, 1, 1, 4, 4)
    frust_h = frust_h.view(1, 1, D, H, W, 4, 1)
    points = (transform @ frust_h).squeeze(-1)  # (B, N, D, H, W, 4)
    return points[..., :3]


def voxel_pool(features, geom, bev_x_range, bev_y_range, bev_res, bev_shape):
    """Sum-pool features at 3D points into a BEV grid.

    features: (B, N, D, H', W', C)
    geom: (B, N, D, H', W', 3) — (X_forward, Y_right, Z_up) in car frame

    Returns: (B, C, H_bev, W_bev).
    """
    B, N, D, H, W, C = features.shape
    H_bev, W_bev = bev_shape

    x = geom[..., 0]
    y = geom[..., 1]

    # X (forward) → row (i); Y (right) shifted by +50 → col (j)
    bx = ((x - bev_x_range[0]) / bev_res).long()
    by = ((y - bev_y_range[0]) / bev_res).long()
    in_grid = (bx >= 0) & (bx < H_bev) & (by >= 0) & (by < W_bev)

    feat_flat = features.reshape(B, -1, C)
    bev_idx = (bx * W_bev + by).reshape(B, -1)
    mask_flat = in_grid.reshape(B, -1)

    feat_flat = feat_flat * mask_flat.unsqueeze(-1).to(feat_flat.dtype)
    bev_idx = torch.where(mask_flat, bev_idx, torch.zeros_like(bev_idx))

    bev = torch.zeros(B, H_bev * W_bev, C, device=features.device, dtype=features.dtype)
    bev.scatter_add_(1, bev_idx.unsqueeze(-1).expand(-1, -1, C), feat_flat)
    return bev.view(B, H_bev, W_bev, C).permute(0, 3, 1, 2).contiguous()


class CamEncoder(nn.Module):
    """ResNet18 → 1x1 head producing (D depth weights, C features) per pixel."""

    def __init__(self, depth_bins=DEPTH_BINS, feat_dim=64, pretrained=True):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = torchvision.models.resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # to layer4: /32, 512 ch
        self.depth_bins = depth_bins
        self.feat_dim = feat_dim
        self.head = nn.Conv2d(512, depth_bins + feat_dim, 1)

    def forward(self, x):
        # x: (B*N, 3, H, W) → volume (B*N, D, C, H', W')
        feat = self.backbone(x)
        head = self.head(feat)
        depth = head[:, : self.depth_bins].softmax(dim=1)
        feat_out = head[:, self.depth_bins:]
        return depth.unsqueeze(2) * feat_out.unsqueeze(1)


class BEVDecoder(nn.Module):
    def __init__(self, in_ch=64, out_ch=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_ch, 1),
        )

    def forward(self, x):
        return self.net(x)


class LiftSplatShoot(nn.Module):
    def __init__(
        self,
        image_size=IMAGE_SIZE,
        downsample=DOWNSAMPLE,
        depth_range=DEPTH_RANGE,
        depth_bins=DEPTH_BINS,
        feat_dim=64,
        bev_x_range=BEV_X_RANGE,
        bev_y_range=BEV_Y_RANGE,
        bev_res=BEV_RES,
        bev_shape=OUT_SIZE,
        pretrained=True,
    ):
        super().__init__()
        self.image_size = image_size
        self.depth_bins = depth_bins
        self.feat_dim = feat_dim
        self.bev_x_range = bev_x_range
        self.bev_y_range = bev_y_range
        self.bev_res = bev_res
        self.bev_shape = bev_shape

        self.encoder = CamEncoder(depth_bins=depth_bins, feat_dim=feat_dim, pretrained=pretrained)
        self.bev_decoder = BEVDecoder(in_ch=feat_dim, out_ch=1)

        self.register_buffer(
            "frustum",
            make_frustum(image_size, downsample, depth_range, depth_bins),
            persistent=False,
        )

    def forward(self, images, intrinsics, car2cams):
        """
        images: (B, N, 3, H, W)
        intrinsics: (B, N, 3, 3) — K mapping car-frame point to image pixel
        car2cams: (B, N, 4, 4) — camera→car transform
        """
        B, N, C, H, W = images.shape
        x = images.view(B * N, C, H, W)
        volume = self.encoder(x)  # (B*N, D, C_feat, H', W')
        _, D, Cf, Hf, Wf = volume.shape
        volume = volume.view(B, N, D, Cf, Hf, Wf).permute(0, 1, 2, 4, 5, 3).contiguous()
        # volume: (B, N, D, H', W', C_feat)

        geom = get_geometry(self.frustum, intrinsics, car2cams)  # (B, N, D, H', W', 3)
        bev = voxel_pool(volume, geom, self.bev_x_range, self.bev_y_range, self.bev_res, self.bev_shape)
        return self.bev_decoder(bev)
