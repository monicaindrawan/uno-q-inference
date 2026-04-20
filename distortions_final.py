"""
distortions_final.py
====================
Consolidated distortion library for the Node Learning training pipeline.

Contains only the distortions actively used by the 4-node system
(IR, colour_shifted, cheap, normal) plus the random environmental
transforms applied on top (stereo, occlusion, angle, zoom, overcast).

See distortions.py / distortions_v2.py / distortions_v3.py for the full
catalogue of available distortions.

Input convention (all functions):
  np.ndarray (H, W, 3) uint8 RGB
"""

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


# ============================================================================
# Shared Utilities
# ============================================================================

def _validate_input(image: np.ndarray) -> np.ndarray:
    img = image.copy()
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.ndim == 3 and img.shape[2] == 1:
        img = np.concatenate([img] * 3, axis=-1)
    if img.dtype in (np.float32, np.float64):
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


# ============================================================================
# Camera Distortions (one per node)
# ============================================================================

def cheap_sensor(image: np.ndarray, scale_factor=None, noise_std=None) -> np.ndarray:
    """
    Simulates budget camera: low-res sensor blur + moderate noise.
    Example: doorbell cam, cheap IP camera.
    """
    image = _validate_input(image)
    if scale_factor is None:
        scale_factor = np.random.uniform(0.35, 0.55)
    if noise_std is None:
        noise_std = np.random.uniform(8, 18)
    h, w = image.shape[:2]
    pil_img = Image.fromarray(image)
    small = pil_img.resize((int(w * scale_factor), int(h * scale_factor)), Image.BILINEAR)
    img = np.array(small.resize((w, h), Image.BILINEAR)).astype(np.float32)
    return np.clip(img + np.random.normal(0, noise_std, img.shape), 0, 255).astype(np.uint8)


def grayscale_ir(image: np.ndarray, ir_weighted=None, noise_std=None) -> np.ndarray:
    """
    Simulates B&W security camera or IR camera in night mode.
    Uses IR-weighted (0.5,0.4,0.1) or standard luminance (0.299,0.587,0.114) conversion.
    """
    image = _validate_input(image)
    if ir_weighted is None:
        ir_weighted = np.random.random() < 0.5
    if noise_std is None:
        noise_std = np.random.uniform(5, 15)
    img = image.astype(np.float32)
    weights = np.array([0.5, 0.4, 0.1]) if ir_weighted else np.array([0.299, 0.587, 0.114])
    gray = np.dot(img, weights)
    gray = np.clip(gray + np.random.normal(0, noise_std, gray.shape), 0, 255)
    return np.stack([gray] * 3, axis=-1).astype(np.uint8)


def colour_shifted(image: np.ndarray, preset=None, r_gain=None, g_gain=None, b_gain=None) -> np.ndarray:
    """
    Simulates incorrect white balance or unusual lighting (sodium lamps, fluorescent, tungsten).
    """
    image = _validate_input(image)
    presets = {
        'sodium': (1.3, 1.05, 0.7),
        'fluorescent': (0.85, 1.1, 0.95),
        'tungsten': (1.2, 1.0, 0.75),
    }
    if preset is None:
        preset = np.random.choice(list(presets.keys())) if np.random.random() < 0.5 else 'random'
    if preset == 'random':
        r_gain = r_gain or np.random.uniform(0.8, 1.3)
        g_gain = g_gain or np.random.uniform(0.9, 1.1)
        b_gain = b_gain or np.random.uniform(0.7, 1.2)
    else:
        base = presets[preset]
        r_gain = base[0] * np.random.uniform(0.95, 1.05)
        g_gain = base[1] * np.random.uniform(0.98, 1.02)
        b_gain = base[2] * np.random.uniform(0.95, 1.05)
    img = image.astype(np.float32)
    img[:, :, 0] *= r_gain
    img[:, :, 1] *= g_gain
    img[:, :, 2] *= b_gain
    return np.clip(img, 0, 255).astype(np.uint8)


def normal_camera(image: np.ndarray) -> np.ndarray:
    """
    Identity distortion for a standard vehicle-mounted camera.
    No image degradation applied.
    """
    return _validate_input(image)


# ============================================================================
# Random Environmental Transforms
# Applied with independent probabilities on top of camera distortions.
# Order: stereo -> angle -> zoom -> lighting -> occlusion
# ============================================================================

def stereo_shift(image: np.ndarray, disparity: int = None,
                 direction: str = None, border_mode: str = "reflect") -> np.ndarray:
    """
    Simulates the view from one camera of a stereo pair by introducing
    horizontal pixel disparity.

    Args:
        disparity: Horizontal shift in pixels (10-25). Randomised if None.
        direction: "left", "right", or None (random). Controls shift direction.
        border_mode: How to fill exposed borders. One of:
            "reflect" — mirror padding (default, most natural)
            "replicate" — repeat edge pixels
            "wrap" — wrap around
            "zero" — black fill (original behaviour)
    """
    image = _validate_input(image)

    if disparity is None:
        disparity = np.random.randint(10, 15)

    # Apply direction
    if direction == "left":
        disparity = abs(disparity)
    elif direction == "right":
        disparity = -abs(disparity)
    elif direction is None:
        disparity = disparity * np.random.choice([-1, 1])

    if _CV2_AVAILABLE:
        border_map = {
            "reflect": cv2.BORDER_REFLECT_101,
            "replicate": cv2.BORDER_REPLICATE,
            "wrap": cv2.BORDER_WRAP,
            "zero": cv2.BORDER_CONSTANT,
        }
        h, w = image.shape[:2]
        M = np.float32([[1, 0, disparity], [0, 1, 0]])
        return cv2.warpAffine(image, M, (w, h),
                              borderMode=border_map.get(border_mode, cv2.BORDER_REFLECT_101))
    else:
        # Fallback: numpy with reflect padding
        abs_d = abs(disparity)
        if border_mode == "zero":
            result = np.zeros_like(image)
            if disparity > 0:
                result[:, disparity:] = image[:, :-disparity]
            else:
                result[:, :disparity] = image[:, abs_d:]
            return result
        else:
            # Reflect pad horizontally, then slice
            padded = np.pad(image, ((0, 0), (abs_d, abs_d), (0, 0)), mode="reflect")
            if disparity > 0:
                return padded[:, :image.shape[1], :]
            else:
                return padded[:, 2 * abs_d:2 * abs_d + image.shape[1], :]


def horizontal_sign_angle(image: np.ndarray, angle_deg: float = None) -> np.ndarray:
    """
    Simulates a camera positioned to the LEFT or RIGHT of a road sign.

    Args:
        angle_deg: Horizontal viewing angle (±10 to ±40). Randomised if None.
    """
    image = _validate_input(image)
    h, w = image.shape[:2]

    if angle_deg is None:
        angle_deg = np.random.uniform(10, 40) * np.random.choice([-1, 1])

    rad = np.deg2rad(abs(angle_deg))
    cos_a = np.cos(rad)
    new_w_near = int(w * cos_a)
    offset = (w - new_w_near) // 2

    if _CV2_AVAILABLE:
        if angle_deg > 0:
            src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            dst = np.float32([
                [offset, 0], [w, 0], [w, h], [offset, h],
            ])
        else:
            src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            dst = np.float32([
                [0, 0], [w - offset, 0], [w - offset, h], [0, h],
            ])
        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(image, M, (w, h))
    else:
        shear = np.tan(rad) * 0.3 * np.sign(angle_deg)
        return np.array(Image.fromarray(image).transform(
            (w, h), Image.AFFINE,
            (1, shear, -shear * h / 2, 0, 1, 0),
            resample=Image.BILINEAR,
        ))


def vertical_sign_angle(image: np.ndarray, angle_deg: float = None) -> np.ndarray:
    """
    Simulates a camera positioned ABOVE or BELOW a road sign.

    Args:
        angle_deg: Vertical viewing angle (±10 to ±40). Randomised if None.
    """
    image = _validate_input(image)
    h, w = image.shape[:2]

    if angle_deg is None:
        angle_deg = np.random.uniform(10, 40) * np.random.choice([-1, 1])

    rad = np.deg2rad(abs(angle_deg))
    cos_a = np.cos(rad)
    new_h_near = int(h * cos_a)
    offset = (h - new_h_near) // 2

    if _CV2_AVAILABLE:
        if angle_deg > 0:
            src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            dst = np.float32([
                [0, offset], [w, offset], [w, h], [0, h],
            ])
        else:
            src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            dst = np.float32([
                [0, 0], [w, 0], [w, h - offset], [0, h - offset],
            ])
        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(image, M, (w, h))
    else:
        shear = np.tan(rad) * 0.3 * np.sign(angle_deg)
        return np.array(Image.fromarray(image).transform(
            (w, h), Image.AFFINE,
            (1, 0, 0, shear, 1, -shear * w / 2),
            resample=Image.BILINEAR,
        ))


def sign_approaching(image: np.ndarray, zoom_factor: float = None,
                     tilt_deg: float = None) -> np.ndarray:
    """
    Simulates a vehicle approaching a road sign head-on (zoom in + upward tilt).

    Args:
        zoom_factor: How much to zoom in (1.05-1.25). Randomised if None.
        tilt_deg: Slight upward perspective tilt in degrees (5-20). Randomised if None.
    """
    image = _validate_input(image)
    h, w = image.shape[:2]

    if zoom_factor is None:
        zoom_factor = np.random.uniform(1.05, 1.25)
    if tilt_deg is None:
        tilt_deg = np.random.uniform(5, 20)

    if _CV2_AVAILABLE:
        cx, cy = w / 2, h / 2
        M_zoom = cv2.getRotationMatrix2D((cx, cy), 0, zoom_factor)
        zoomed = cv2.warpAffine(image, M_zoom, (w, h))

        rad = np.deg2rad(tilt_deg)
        margin = int(h * np.sin(rad) * 0.4)
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([
            [margin, 0],
            [w - margin, 0],
            [w, h],
            [0, h],
        ])
        M_tilt = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(zoomed, M_tilt, (w, h))
    else:
        crop_margin = int(min(h, w) * (1 - 1 / zoom_factor) / 2)
        cropped = image[crop_margin:h - crop_margin, crop_margin:w - crop_margin]
        pil_img = Image.fromarray(cropped).resize((w, h), Image.BILINEAR)
        return np.array(pil_img)


def overcast_flat_light(image: np.ndarray,
                        desaturation: float = None,
                        brightness: float = None) -> np.ndarray:
    """
    Simulates overcast sky diffuse lighting (flat, desaturated colours).

    Args:
        desaturation: How much to desaturate (0.1-0.5). Randomised if None.
        brightness: Overall brightness adjustment (0.85-1.1). Randomised if None.
    """
    image = _validate_input(image)

    if desaturation is None:
        desaturation = np.random.uniform(0.1, 0.5)
    if brightness is None:
        brightness = np.random.uniform(0.85, 1.1)

    img = image.astype(np.float32)

    gray = (0.299 * img[:, :, 0] +
            0.587 * img[:, :, 1] +
            0.114 * img[:, :, 2])[:, :, np.newaxis]
    img = img * (1 - desaturation) + gray * desaturation
    img = img * brightness

    return np.clip(img, 0, 255).astype(np.uint8)


def random_occlusion(image: np.ndarray, coverage: float = None) -> np.ndarray:
    """
    Masks 30-50% of the image with an irregular blob simulating a foreground
    object blocking part of the camera's view.

    The fill colour is grayscale-aware: if the image is monochrome (e.g. IR),
    the occluder is also drawn in grayscale.

    Args:
        coverage: Fraction of image to occlude (0.3-0.5). Randomised if None.
    """
    image = _validate_input(image)

    if coverage is None:
        coverage = np.random.uniform(0.30, 0.50)

    h, w = image.shape[:2]
    result = image.copy()

    # Detect if image is grayscale (all channels equal) and pick appropriate fill
    is_gray = np.allclose(image[:, :, 0], image[:, :, 1]) and np.allclose(image[:, :, 1], image[:, :, 2])
    if is_gray:
        v = int(np.random.randint(0, 60))
        fill = [v, v, v]
    else:
        fill = np.random.randint(0, 60, size=3).tolist()

    cx = np.random.randint(int(w * 0.1), int(w * 0.9))
    cy = np.random.randint(int(h * 0.1), int(h * 0.9))

    target_area = coverage * h * w
    base_r = np.sqrt(target_area / np.pi)
    rx = base_r * np.random.uniform(0.8, 1.8)
    ry = base_r * np.random.uniform(0.8, 1.8)

    n_pts = np.random.randint(12, 20)
    angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    radii_x = rx * np.random.uniform(0.6, 1.4, n_pts)
    radii_y = ry * np.random.uniform(0.6, 1.4, n_pts)
    pts_x = (cx + radii_x * np.cos(angles)).astype(np.int32)
    pts_y = (cy + radii_y * np.sin(angles)).astype(np.int32)
    pts = np.stack([pts_x, pts_y], axis=1)

    if _CV2_AVAILABLE:
        cv2.fillPoly(result, [pts], color=fill)
    else:
        # Rasterise polygon with PIL ImageDraw
        from PIL import ImageDraw
        pil_img = Image.fromarray(result)
        draw = ImageDraw.Draw(pil_img)
        poly = [(int(x), int(y)) for x, y in zip(pts_x, pts_y)]
        draw.polygon(poly, fill=tuple(fill))
        result = np.array(pil_img)

    return result


# ============================================================================
# Transform Pipeline Classes
# ============================================================================

def _fill_black_borders(image: np.ndarray, threshold: int = 5) -> np.ndarray:
    """
    Detect pure-black pixels introduced by geometric warps at image borders
    and inpaint them. Only fills black regions that are connected to an edge
    of the image, so dark image content in the interior is left untouched.
    """
    black = (image.max(axis=2) <= threshold).astype(np.uint8)
    if black.sum() == 0:
        return image

    # Flood-fill from each edge to find border-connected black regions only
    h, w = black.shape
    mask = np.zeros_like(black)
    # Check all border pixels; if black, flood-fill to mark the connected region
    if _CV2_AVAILABLE:
        # cv2.floodFill needs a mask 2px larger than image
        ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        border_pixels = set()
        for x in range(w):
            border_pixels.add((x, 0))
            border_pixels.add((x, h - 1))
        for y in range(h):
            border_pixels.add((0, y))
            border_pixels.add((w - 1, y))
        for (x, y) in border_pixels:
            if black[y, x] == 1 and ff_mask[y + 1, x + 1] == 0:
                cv2.floodFill(black, ff_mask, (x, y), 255)
        mask = (ff_mask[1:-1, 1:-1] > 0).astype(np.uint8)

        if mask.sum() == 0:
            return image
        return cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    else:
        # Fallback: only fill black pixels in the outermost N-pixel border
        border_width = max(1, min(h, w) // 8)
        mask[:border_width, :] = black[:border_width, :]
        mask[-border_width:, :] = black[-border_width:, :]
        mask[:, :border_width] = black[:, :border_width]
        mask[:, -border_width:] = black[:, -border_width:]
        if mask.sum() == 0:
            return image
        non_black = image[mask == 0]
        if len(non_black) == 0:
            return image
        mean_colour = non_black.mean(axis=0).astype(np.uint8)
        result = image.copy()
        result[mask == 1] = mean_colour
        return result


class RandomTransformChain:
    """
    Applies a sequence of numpy-based transforms, each with an independent probability.
    After all transforms, any pure-black border pixels left by geometric warps
    are filled using inpainting.

    Args:
        transforms: List of (transform_fn, probability, kwargs_dict) tuples.
                    Each transform_fn takes np.ndarray and returns np.ndarray.
    """
    def __init__(self, transforms=None):
        self.transforms = transforms or []

    def __call__(self, image: np.ndarray) -> np.ndarray:
        for fn, prob, kwargs in self.transforms:
            if np.random.random() < prob:
                image = fn(image, **kwargs)
        return _fill_black_borders(image)


CAMERA_PRESETS = {
    "vehicle_mounted": {
        "crop_scale": (0.7, 1.0),
        "enable_hflip": True,
        "affine_degrees": 15,
        "affine_scale": (0.85, 1.15),
        "affine_shear": 10,
        "perspective_distortion": 0.3,
        "perspective_p": 0.5,
    },
    "pole_fixed": {
        "crop_scale": (0.9, 1.0),
        "enable_hflip": False,
        "affine_degrees": 3,
        "affine_scale": (0.95, 1.05),
        "affine_shear": 2,
        "perspective_distortion": 0.0,
        "perspective_p": 0.0,
    },
    "handheld": {
        "crop_scale": (0.8, 1.0),
        "enable_hflip": True,
        "affine_degrees": 8,
        "affine_scale": (0.9, 1.1),
        "affine_shear": 5,
        "perspective_distortion": 0.15,
        "perspective_p": 0.3,
    },
    "ptz_camera": {
        "crop_scale": (0.75, 1.0),
        "enable_hflip": False,
        "affine_degrees": 10,
        "affine_scale": (0.85, 1.15),
        "affine_shear": 5,
        "perspective_distortion": 0.2,
        "perspective_p": 0.4,
    },
}


class NodeTransform:
    """
    Unified transform wrapping a numpy distortion + random environmental
    transforms + torchvision geometric augmentation.

    Pipeline: Camera distortion (numpy) -> Random transforms (numpy)
              -> Geometric augmentation (torchvision) -> Tensor
    """
    def __init__(self, distortion_fn=None, preset="pole_fixed", output_size=None,
                 training=True, random_transforms=None,
                 crop_scale=None, enable_hflip=None,
                 affine_degrees=None, affine_scale=None, affine_shear=None,
                 perspective_distortion=None, perspective_p=None, fill_value=0):
        if preset not in CAMERA_PRESETS:
            raise ValueError(f"Unknown preset '{preset}'. Available: {list(CAMERA_PRESETS.keys())}")
        self.distortion_fn = distortion_fn
        self.preset = preset
        self.output_size = output_size
        self.training = training
        self.random_transforms = random_transforms
        self.fill_value = fill_value
        cfg = CAMERA_PRESETS[preset].copy()
        self.crop_scale = crop_scale or cfg["crop_scale"]
        self.enable_hflip = enable_hflip if enable_hflip is not None else cfg["enable_hflip"]
        self.affine_degrees = affine_degrees or cfg["affine_degrees"]
        self.affine_scale = affine_scale or cfg["affine_scale"]
        self.affine_shear = affine_shear or cfg["affine_shear"]
        self.perspective_distortion = perspective_distortion if perspective_distortion is not None else cfg["perspective_distortion"]
        self.perspective_p = perspective_p if perspective_p is not None else cfg["perspective_p"]
        self._train_pipeline = self._eval_pipeline = self._cached_size = None

    def _build_train_pipeline(self, size):
        trs = [T.RandomResizedCrop(size=size, scale=self.crop_scale, ratio=(0.95, 1.05))]
        if self.enable_hflip:
            trs.append(T.RandomHorizontalFlip(p=0.5))
        trs.append(T.RandomAffine(degrees=self.affine_degrees, scale=self.affine_scale,
                                   shear=self.affine_shear, fill=self.fill_value))
        if self.perspective_p > 0:
            trs.append(T.RandomPerspective(distortion_scale=self.perspective_distortion,
                                            p=self.perspective_p, fill=self.fill_value))
        trs.append(T.ToTensor())
        return T.Compose(trs)

    def _build_eval_pipeline(self, size):
        return T.Compose([T.Resize(size), T.ToTensor()])

    def _get_pipeline(self, size):
        if self._cached_size != size:
            self._train_pipeline = self._build_train_pipeline(size)
            self._eval_pipeline = self._build_eval_pipeline(size)
            self._cached_size = size
        return self._train_pipeline if self.training else self._eval_pipeline

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        if self.distortion_fn is not None:
            image = self.distortion_fn(image)
        if self.random_transforms is not None:
            image = self.random_transforms(image)
        pil_img = Image.fromarray(image)
        size = self.output_size if self.output_size is not None else min(image.shape[:2])
        return self._get_pipeline(size)(pil_img)

    def eval(self):
        self.training = False
        return self

    def train(self):
        self.training = True
        return self
