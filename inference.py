"""
Inference — Node Inference Engine
==================================

Loads the per-node model and distortion pipeline at import time, then exposes
three inference functions called by web_app.py:

    solo_inference()          — classify using only this node's CNN
    fusion_inference()        — always request peer embedding and fuse
    collaborative_inference() — use solo unless confidence < threshold,
                                then fall back to fusion

Peer communication uses Bluetooth RFCOMM (via PeerNode). Two message types:
    req_emb  — send a resized PNG to the peer and ask for its embedding
    emb      — receive the peer's embedding + confidence as base64 JSON

The active node profile (ir or colour_shifted) is set via the NODE_NAME
environment variable, which selects the model weights, distortion function,
and augmentation chain.
"""

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
from merge_operators import ConfidenceWeightedMean, RobustMedian, TopKConfident
from node_model import NodeModel
from bluetooth_peer_node import PeerNode, load_peer_macs

NODE_NAME = os.environ.get("NODE_NAME", "ir")
CONFIDENCE_THRESHOLD = 0.6  # below this, collaborative mode triggers fusion


# =============================================================================
# Config
# Per-node model paths, distortion functions, and augmentation probabilities.
# NODE_NAME selects the active profile at startup.
# =============================================================================

def _build_merge_op(op: str | None):
    """Instantiate a merge operator from its name. Returns None for fusion_head
    (the model's built-in cross-attention path is used instead)."""
    if op == "confidence_weighted_mean":
        return ConfidenceWeightedMean()
    if op == "robust_median":
        return RobustMedian()
    if op == "top_k_confident":
        return TopKConfident()
    if op == "fusion_head":
        return None
    raise ValueError(f"Unknown merge operator: {op!r}")

DEVICE = "cpu"

# Maps node name to its trained model checkpoint
MODEL_PATHS = {
    "ir": "models/gtsrb_ir_2node_with_fusion.pt",
    "colour_shifted": "models/gtsrb_colour_shifted_2node_with_fusion.pt"
}

# Maps node name to its distortion function and geometric preset
NODE_CONFIGS = {
    "ir": {"distortion_fn": grayscale_ir, "preset": "pole_fixed", "size": 48},
    "colour_shifted": {"distortion_fn": colour_shifted, "preset": "vehicle_mounted", "size": 48},
}

IMG_SIZE = 48

# Per-node random augmentation chain: (transform_fn, probability, kwargs)
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


# =============================================================================
# Model & Transform Initialisation
# Loaded once at import time and reused for every request.
# =============================================================================

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


# =============================================================================
# Bluetooth Peer Embedding Exchange
# _peer_emb_queue holds at most one pending embedding from the peer.
# _on_peer_message() handles two message types:
#   req_emb — peer wants our embedding; compute and send it back
#   emb     — peer's embedding arriving in response to our request
# =============================================================================

_peer_emb_queue: queue.Queue[tuple] = queue.Queue(maxsize=1)


def _embedding_to_msg(emb: torch.Tensor, confidence: float) -> str:
    """Serialise an embedding tensor to a base64 JSON string for transmission."""
    arr = emb.detach().cpu().numpy().astype(np.float32)
    return json.dumps({
        "type": "emb",
        "shape": list(arr.shape),
        "data": base64.b64encode(arr.tobytes()).decode(),
        "confidence": confidence,
    })


def _on_peer_message(text: str) -> None:
    """Dispatch an incoming Bluetooth message to the appropriate handler."""
    try:
        obj = json.loads(text)
        msg_type = obj.get("type")

        if msg_type == "req_emb":
            print("Received req_emb message")
            # Peer sent us a resized image — compute our embedding and reply
            img_bytes = base64.b64decode(obj["data"])
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            with torch.no_grad():
                image_tensor = transform_image(image).to(DEVICE)
                emb = image_to_embedding(image_tensor)
                logits, _ = model(image_tensor)
                confidence = confidence_entropy(logits)
            peer_node.send(_embedding_to_msg(emb, confidence))

        elif msg_type == "emb":
            print("Received emb message")
            # Decode the peer's embedding and put it in the queue for fusion
            arr = np.frombuffer(base64.b64decode(obj["data"]), dtype=np.float32).copy()
            emb = torch.from_numpy(arr.reshape(obj["shape"])).to(DEVICE)
            conf = float(obj.get("confidence", 0.0))
            try:
                _peer_emb_queue.get_nowait()  # discard any stale value
            except queue.Empty:
                pass
            _peer_emb_queue.put((emb, conf))

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
    """Return connection state and message counters for the UI status bar."""
    return {
        "connected": peer_node.is_connected,
        "active_peer_mac": peer_node.active_peer_mac,
        "candidate_peer_macs": peer_node.peer_macs,
        "messages_sent": peer_node.messages_sent,
        "messages_received": peer_node.messages_received,
        "messages_exchanged": peer_node.messages_exchanged,
    }


# =============================================================================
# Image Preprocessing Helpers
# =============================================================================

def bytes_to_resized_image(image_bytes: bytes) -> Image:
    """Decode raw bytes to a PIL RGB image resized to IMG_SIZE x IMG_SIZE."""
    return Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(
        (IMG_SIZE, IMG_SIZE), Image.BILINEAR)


def transform_image(image: Image) -> torch.Tensor:
    """Apply the node's distortion + augmentation pipeline; add batch dim."""
    return transform(np.array(image)).unsqueeze(0)


def image_to_embedding(image: torch.Tensor):
    """Run the CNN encoder and return the embedding vector (no classifier)."""
    return model.encoder(image)


# =============================================================================
# Inference Functions
# =============================================================================

def solo_inference(image_bytes: bytes) -> int:
    """Classify using only this node's CNN — no peer communication."""
    image = bytes_to_resized_image(image_bytes)
    image_tensor = transform_image(image).to(DEVICE)
    with torch.no_grad():
        logits, _ = model(image_tensor)
    return {
        "pred_class": logits.argmax(1).item(),
        "method": "solo_inference"
    }


def fusion_inference(image_bytes: bytes, merge_operator: str | None = None, timeout: float = 5.0) -> int:
    """Always fuse with the peer — request its embedding and combine.

    Falls back to solo if the peer does not respond within timeout seconds.
    merge_operator selects how embeddings are combined:
        None / "fusion_head" — learned cross-attention (model.fused_forward)
        otherwise            — a statistical operator (confidence weighted, etc.)
    """
    image = bytes_to_resized_image(image_bytes)
    with torch.no_grad():
        image_tensor = transform_image(image)
        own_emb = image_to_embedding(image_tensor)
        logits, _ = model(image_tensor)
        confidence = confidence_entropy(logits)

    # Send resized image bytes to peer so they can compute embedding with their pipeline
    resized_buf = io.BytesIO()
    image.save(resized_buf, format="PNG")
    peer_node.send(json.dumps({
        "type": "req_emb",
        "data": base64.b64encode(resized_buf.getvalue()).decode(),
    }))

    try:
        peer_emb, peer_conf = _peer_emb_queue.get(timeout=timeout)
    except queue.Empty:
        print("No peer embedding received within timeout")
        return {
            "pred_class": solo_inference(image_bytes)["pred_class"],
            "method": "solo_inference",
            "reason": "fallback_peer_no_response"
        }

    with torch.no_grad():
        merge_op = _build_merge_op(merge_operator)
        if merge_op is None:
            logits, _ = model.fused_forward(own_emb, [peer_emb])
        else:
            peer_contexts = {
                "confidences": [peer_conf],
                "requesting_confidence": confidence,
            }
            logits = model.classifier(merge_op.merge(own_emb, [peer_emb], peer_contexts))

    return {
        "pred_class": logits.argmax(1).item(),
        "method": "fusion_inference"
    }


def confidence_entropy(logits: torch.Tensor) -> float:
    """Compute normalised confidence as 1 - (entropy / max_entropy).

    Returns a value in [0, 1]; higher means more confident.
    Uses normalised entropy so the score is comparable across class counts.
    """
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-8)
    entropy = -(probs * log_probs).sum(dim=-1)
    max_entropy = np.log(probs.size(-1))
    return (1.0 - entropy / max_entropy).item()


def collaborative_inference(image_bytes: bytes, merge_operator: str | None = None, timeout: float = 5.0) -> dict:
    """Use solo inference if confident; escalate to fusion if not.

    Decision boundary: confidence_entropy(logits) >= CONFIDENCE_THRESHOLD
    Falls back to solo (with reason tag) if the peer does not respond in time.
    extra_info["solo_pred_class"] is included so the evaluator can track
    cases where fusion fixed or broke the solo prediction.
    """
    image = bytes_to_resized_image(image_bytes)
    with torch.no_grad():
        image_tensor = transform_image(image).to(DEVICE)
        logits, _ = model(image_tensor)

    confidence = confidence_entropy(logits)

    # High confidence — skip peer communication entirely
    if confidence >= CONFIDENCE_THRESHOLD:
        return {
            "pred_class": logits.argmax(1).item(),
            "method": "solo_inference",
            "reason": "high_confidence",
            "confidence": confidence,
        }

    # Low confidence — request peer embedding and fuse
    own_emb = image_to_embedding(image_tensor)

    resized_buf = io.BytesIO()
    image.save(resized_buf, format="PNG")
    peer_node.send(json.dumps({
        "type": "req_emb",
        "data": base64.b64encode(resized_buf.getvalue()).decode(),
    }))

    try:
        peer_emb, peer_conf = _peer_emb_queue.get(timeout=timeout)
    except queue.Empty:
        print("No peer embedding received within timeout")
        return {
            "pred_class": logits.argmax(1).item(),
            "method": "solo_inference",
            "reason": "fallback_peer_no_response",
            "confidence": confidence
        }

    with torch.no_grad():
        merge_op = _build_merge_op(merge_operator)
        if merge_op is None:
            fused_logits, _ = model.fused_forward(own_emb, [peer_emb])
        else:
            peer_contexts = {
                "confidences": [peer_conf],
                "requesting_confidence": confidence,
            }
            fused_logits = model.classifier(merge_op.merge(own_emb, [peer_emb], peer_contexts))

    return {
        "pred_class": fused_logits.argmax(1).item(),
        "method": "fusion_inference",
        "reason": "low_confidence",
        "confidence": confidence,
        "extra_info": {
            "solo_pred_class": logits.argmax(1).item(),
        }
    }
