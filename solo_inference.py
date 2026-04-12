import io
import torch
import numpy as np
from PIL import Image

from distortions_final import RandomTransformChain, colour_shifted, grayscale_ir, NodeTransform, horizontal_sign_angle, overcast_flat_light, random_occlusion, sign_approaching, stereo_shift, vertical_sign_angle
from node_model import NodeModel

MODEL_PATH = "models/gtsrb_ir_2node_with_fusion.pt"
NODE_NAME = "ir"

DEVICE = "cpu"
NODE_CONFIGS = {
    "ir": {"distortion_fn": grayscale_ir, "preset": "pole_fixed", "size": 48},
    "colour_shifted": {"distortion_fn": colour_shifted, "preset": "vehicle_mounted", "size": 48},
}
RANDOM_TRANSFORMS = {
    "ir": [
        (stereo_shift,          0.3,  {"border_mode": "reflect"}),
        (horizontal_sign_angle, 0.2,  {}),
        (vertical_sign_angle,   0.2,  {}),
        (sign_approaching,      0.1,  {}),
        (overcast_flat_light,   0.1,  {}),
        (random_occlusion,      0.2,  {}),
    ],
    "colour_shifted": [
        (stereo_shift,          0.15, {"border_mode": "reflect"}),
        (horizontal_sign_angle, 0.3,  {}),
        (vertical_sign_angle,   0.3,  {}),
        (sign_approaching,      0.1,  {}),
        (overcast_flat_light,   0.4,  {}),
        (random_occlusion,      0.15, {}),
    ],
}

# Model and transform are initialized once at import time and reused for every request.
model = NodeModel(include_fusion=True)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

random_chain = RandomTransformChain(RANDOM_TRANSFORMS.get(NODE_NAME, []))
cfg = NODE_CONFIGS[NODE_NAME]

transform = NodeTransform(
    distortion_fn=cfg["distortion_fn"],
    preset=cfg["preset"],
    output_size=cfg["size"],
    training=False,
    random_transforms=random_chain,
)

def solo_inference(image_bytes: bytes) -> int:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(
        (cfg["size"], cfg["size"]), Image.BILINEAR)
    tensor = transform(np.array(img)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits, _ = model(tensor)
    return logits.argmax(1).item()