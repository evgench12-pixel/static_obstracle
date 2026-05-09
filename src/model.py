"""Phase A baseline: per-camera ResNet18 + concat fusion + conv decoder.

This model does NOT use camera calibration. It serves as a sanity-check
pipeline for end-to-end training; the real geometric model (LSS) goes in
Phase B.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from src.config import OUT_SIZE


class MultiCamCNN(nn.Module):
    def __init__(self, num_cams=4, out_size=OUT_SIZE, pretrained=True):
        super().__init__()
        self.num_cams = num_cams
        self.out_size = out_size

        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = torchvision.models.resnet18(weights=weights)
        # Drop avgpool + fc to keep spatial feature maps
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])
        feat_ch = 512

        in_ch = feat_ch * num_cams
        self.decoder = nn.Sequential(
            nn.Conv2d(in_ch, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, images, intrinsics=None, car2cams=None):
        """images: (B, num_cams, 3, H, W) → logits (B, 1, 188, 126).

        intrinsics/car2cams accepted to share signature with geometric models
        but unused here — this baseline ignores camera calibration.
        """
        B, N, C, H, W = images.shape
        x = images.view(B * N, C, H, W)
        feats = self.encoder(x)  # (B*N, 512, h, w)
        _, Cf, Hf, Wf = feats.shape
        feats = feats.view(B, N, Cf, Hf, Wf)
        feats = feats.permute(0, 2, 1, 3, 4).contiguous()
        feats = feats.view(B, N * Cf, Hf, Wf)
        x = self.decoder(feats)
        x = F.interpolate(x, size=self.out_size, mode="bilinear", align_corners=False)
        return x
