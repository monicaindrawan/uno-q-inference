import io
import json
import base64
import queue
import torch
import numpy as np
from PIL import Image
import os
import socket

from distortions_final import RandomTransformChain, colour_shifted, grayscale_ir, NodeTransform, horizontal_sign_angle, overcast_flat_light, random_occlusion, sign_approaching, stereo_shift, vertical_sign_angle
from node_model import NodeModel
from bluetooth_peer_node import PeerNode, load_peer_macs

NODE_NAME = os.environ.get("NODE_NAME", "ir")

DEVICE = "cpu"
MODEL_PATHS = {
    "ir": "models/gtsrb_ir_2node_with_fusion.pt",
    "colour_shifted": "models/gtsrb_colour_shifted_2node_with_fusion.pt"
}
NODE_CONFIGS = {
    "ir": {"distortion_fn": grayscale_ir, "preset": "pole_fixed", "size": 48},
    "colour_shifted": {"distortion_fn": colour_shifted, "preset": "vehicle_mounted", "size": 48},
}
IMG_SIZE = 48
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
print(f"Loading model for node {NODE_NAME} from {MODEL_PATHS[NODE_NAME]}")
model = NodeModel(include_fusion=True)
model.load_state_dict(torch.load(MODEL_PATHS.get(NODE_NAME), map_location=DEVICE))
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

# -- Peer node embedding via Bluetooth --------------------------------------

_peer_emb_queue: queue.Queue[torch.Tensor] = queue.Queue(maxsize=1)


def _embedding_to_msg(emb: torch.Tensor) -> str:
    arr = emb.detach().cpu().numpy().astype(np.float32)
    return json.dumps({
        "type": "emb",
        "shape": list(arr.shape),
        "data": base64.b64encode(arr.tobytes()).decode(),
    })


def _on_peer_message(text: str) -> None:
    try:
        obj = json.loads(text)
        msg_type = obj.get("type")

        if msg_type == "req_emb":
            print("Received req_emb message")
            # Peer sent us already-resized image bytes — compute embedding and send it back.
            img_bytes = base64.b64decode(obj["data"])
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            with torch.no_grad():
                image_tensor = transform_image(image).to(DEVICE)
                emb = image_to_embedding(image_tensor)
            peer_node.send(_embedding_to_msg(emb))

        elif msg_type == "emb":
            print("Received emb message")
            arr = np.frombuffer(base64.b64decode(obj["data"]), dtype=np.float32).copy()
            emb = torch.from_numpy(arr.reshape(obj["shape"])).to(DEVICE)
            try:
                _peer_emb_queue.get_nowait()  # discard any stale value
            except queue.Empty:
                pass
            _peer_emb_queue.put(emb)

    except Exception:
        pass

peer_node = PeerNode(
    my_name=socket.gethostname(),
    peer_macs=load_peer_macs("peers.json"),
    channel=4,
    on_message=_on_peer_message,
)
peer_node.start()

def get_peer_status():
    return {
        "connected": peer_node.is_connected,
        "active_peer_mac": peer_node.active_peer_mac,
        "candidate_peer_macs": peer_node.peer_macs,
        "messages_sent": peer_node.messages_sent,
        "messages_received": peer_node.messages_received,
        "messages_exchanged": peer_node.messages_exchanged,
    }

# ---------------------------------------------------------------------------

def bytes_to_resized_image(image_bytes: bytes) -> Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(
        (IMG_SIZE, IMG_SIZE), Image.BILINEAR)


def transform_image(image: Image) -> torch.Tensor:
    return transform(np.array(image)).unsqueeze(0)


def image_to_embedding(image: torch.Tensor):
    return model.encoder(image)


def solo_inference(image_bytes: bytes) -> int:
    image = bytes_to_resized_image(image_bytes)
    image_tensor = transform_image(image).to(DEVICE)
    with torch.no_grad():
        logits, _ = model(image_tensor)
    return logits.argmax(1).item()


def fusion_inference(image_bytes: bytes, timeout: float = 10.0) -> int:
    image = bytes_to_resized_image(image_bytes)
    with torch.no_grad():
        image_tensor = transform_image(image)
        own_emb = image_to_embedding(image_tensor)

    # Send resized image bytes to peer so they can compute the embedding with their pipeline.
    resized_buf = io.BytesIO()
    image.save(resized_buf, format="PNG")
    peer_node.send(json.dumps({
        "type": "req_emb",
        "data": base64.b64encode(resized_buf.getvalue()).decode(),
    }))

    try:
        peer_emb = _peer_emb_queue.get(timeout=timeout)
    except queue.Empty:
        raise TimeoutError("No peer embedding received within timeout")

    with torch.no_grad():
        logits, _ = model.fused_forward(own_emb, [peer_emb])
    return logits.argmax(1).item()
    pass