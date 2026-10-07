from pathlib import Path
import torch
from .network.DeepLabv3_plus import DeepLabv3Plus

def load_deeplab_from_ckpt(ckpt_path: Path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DeepLabv3Plus(num_classes_det=3).to(device).eval()

    ckpt = torch.load(str(ckpt_path), map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    cleaned = {}
    for k, v in state_dict.items():
        # common patterns from Lightning: 'net.', 'model.'
        if k.startswith("net."):
            cleaned[k[4:]] = v
        elif k.startswith("model."):
            cleaned[k[6:]] = v
        else:
            cleaned[k] = v
    model.load_state_dict(cleaned, strict=False)
    return model

