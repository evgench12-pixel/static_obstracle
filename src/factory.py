"""Model factory — pick architecture by name in CLI."""

from src.lss import LiftSplatShoot
from src.model import MultiCamCNN

MODELS = {
    "multicam_cnn": MultiCamCNN,
    "lss": LiftSplatShoot,                                       # ResNet18 backbone
    "lss_r50": lambda **kw: LiftSplatShoot(backbone="resnet50", **kw),
    "lss_convnext_tiny": lambda **kw: LiftSplatShoot(backbone="convnext_tiny", **kw),
    "lss_convnext_small": lambda **kw: LiftSplatShoot(backbone="convnext_small", **kw),
}


def build_model(name, **kwargs):
    if name not in MODELS:
        raise ValueError(f"Unknown model {name!r}; available: {list(MODELS)}")
    return MODELS[name](**kwargs)
