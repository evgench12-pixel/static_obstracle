"""Model factory — pick architecture by name in CLI."""

from src.lss import LiftSplatShoot
from src.model import MultiCamCNN

MODELS = {
    "multicam_cnn": MultiCamCNN,
    "lss": LiftSplatShoot,
}


def build_model(name, **kwargs):
    if name not in MODELS:
        raise ValueError(f"Unknown model {name!r}; available: {list(MODELS)}")
    return MODELS[name](**kwargs)
