"""Scene classification for keyframes.

Two modes:
1. ONNX CLIP (high quality, needs model download)
2. Color heuristic (always works, zero deps beyond numpy/opencv)
"""

from pathlib import Path
from typing import List, NamedTuple, Optional

import cv2
import numpy as np
from snipclip._cv import imread


class SceneTag(NamedTuple):
    """A scene classification label."""
    label: str        # Chinese scene label
    confidence: float  # 0.0-1.0
    source: str       # "clip" or "heuristic"


# Predefined scene labels in Chinese
SCENE_LABELS = [
    "户外自然风光",
    "山景",
    "海边/水边",
    "城市街景",
    "隧道/洞穴",
    "室内/房间",
    "餐厅/吃饭",
    "车内/驾驶",
    "夜景/暗光",
    "运动/活动",
    "人群/排队",
    "特写/自拍",
]


def classify_frame(
    image_path: Path,
    labels: Optional[List[str]] = None,
) -> List[SceneTag]:
    """Classify a single frame into scene categories.

    Uses ONNX CLIP if available, falls back to color heuristic.

    Args:
        image_path: Path to the frame image.
        labels: List of scene labels to classify against (default: built-in list).

    Returns:
        List of SceneTag, sorted by confidence descending.
    """
    if labels is None:
        labels = SCENE_LABELS

    # Try CLIP first
    clip_tags = _classify_clip(image_path, labels)
    if clip_tags:
        return clip_tags

    # Fall back to heuristic
    return _classify_heuristic(image_path)


# ---- CLIP (ONNX) Implementation ----

_vision_session: Optional = None
_text_session: Optional = None
_clip_tokenizer: Optional = None
_clip_loaded: bool = False  # True if we tried loading (success or failure)


def _load_clip():
    """Load ONNX CLIP vision + text models. Returns (vis_sess, txt_sess, tok) or (None, None, None)."""
    global _vision_session, _text_session, _clip_tokenizer, _clip_loaded

    if _clip_loaded:
        return _vision_session, _text_session, _clip_tokenizer

    _clip_loaded = True

    try:
        import onnxruntime as ort

        model_dir = Path.home() / ".snipclip" / "models" / "clip-vit-b32"
        vis_path = model_dir / "onnx" / "vision_model_quantized.onnx"
        txt_path = model_dir / "onnx" / "text_model_quantized.onnx"

        if not vis_path.exists() or not txt_path.exists():
            return None, None, None

        _vision_session = ort.InferenceSession(str(vis_path), providers=["CPUExecutionProvider"])
        _text_session = ort.InferenceSession(str(txt_path), providers=["CPUExecutionProvider"])

        try:
            from transformers import CLIPTokenizerFast
            _clip_tokenizer = CLIPTokenizerFast.from_pretrained(str(model_dir))
        except ImportError:
            pass

        return _vision_session, _text_session, _clip_tokenizer

    except Exception:
        return None, None, None


def download_clip_model() -> Path:
    """Download CLIP-ViT-B-32 ONNX model from HuggingFace to ~/.snipclip/models/."""
    from huggingface_hub import snapshot_download

    model_dir = Path.home() / ".snipclip" / "models" / "clip-vit-b32"
    vis_path = model_dir / "onnx" / "vision_model_quantized.onnx"

    if vis_path.exists():
        return model_dir

    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading CLIP model to {model_dir}...")
    print("(~600MB, may take a few minutes)")

    snapshot_download(
        "Xenova/clip-vit-base-patch32",
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
        ignore_patterns=["*.bin", "*.safetensors", "*.msgpack", "*.h5"],
    )

    print("CLIP model ready.")
    return model_dir


def _classify_clip(image_path: Path, labels: List[str]) -> Optional[List[SceneTag]]:
    """Classify using ONNX CLIP (separate vision + text encoders). Returns None if model unavailable."""
    vis_sess, txt_sess, tokenizer = _load_clip()
    if vis_sess is None or txt_sess is None or tokenizer is None:
        return None

    try:
        from PIL import Image

        # ---- Image preprocessing (CLIP-ViT-B-32) ----
        img = Image.open(str(image_path)).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img).astype(np.float32) / 255.0
        mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
        img_array = (img_array - mean) / std
        img_array = img_array.transpose(2, 0, 1)[np.newaxis, :]  # [1, 3, 224, 224]

        # ---- Image embedding ----
        vis_input = {vis_sess.get_inputs()[0].name: img_array}
        img_features = vis_sess.run(None, vis_input)[0]
        img_features = img_features / np.linalg.norm(img_features, axis=1, keepdims=True)

        # ---- Text embedding ----
        text_inputs = tokenizer(labels, padding=True, truncation=True, return_tensors="np")
        txt_out = txt_sess.run(None, {
            "input_ids": text_inputs["input_ids"].astype(np.int64),
            "attention_mask": text_inputs["attention_mask"].astype(np.int64),
        })
        text_features = txt_out[0]  # [pooler_output]
        text_features = text_features / np.linalg.norm(text_features, axis=1, keepdims=True)

        # ---- Cosine similarity ----
        similarities = np.dot(img_features, text_features.T)[0]

        tags = []
        for i, label in enumerate(labels):
            tags.append(SceneTag(label=label, confidence=round(float(similarities[i]), 3), source="clip"))

        tags.sort(key=lambda t: t.confidence, reverse=True)
        return tags

    except Exception:
        return None


# ---- Heuristic Fallback ----

def _classify_heuristic(image_path: Path) -> List[SceneTag]:
    """Scene classification based on multi-region visual feature analysis.

    Uses: sky detection, green/blue pixel ratios, edge density,
    regional brightness analysis, and aspect ratio.
    """
    img = imread(image_path)
    if img is None:
        return [SceneTag(label="未知场景", confidence=0.0, source="heuristic")]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]

    # ---- Region masks ----
    sky_region = hsv[0:h//3, :]              # top third
    ground_region = hsv[h//2:, :]             # bottom half
    center_region = hsv[h//4:3*h//4, w//4:3*w//4]  # center

    # ---- Brightness analysis ----
    avg_brightness = float(np.mean(center_region[:, :, 2]))
    sky_brightness = float(np.mean(sky_region[:, :, 2]))
    ground_brightness = float(np.mean(ground_region[:, :, 2]))

    # ---- Color ratios ----
    # Green ratio (nature/foliage): hue 35-85, saturation > 40
    hue = ground_region[:, :, 0]
    sat = ground_region[:, :, 1]
    green_mask = (hue > 35) & (hue < 85) & (sat > 40)
    green_ratio = float(np.mean(green_mask))

    # Blue ratio (water/sky): hue 90-130, saturation > 30
    blue_mask = (hue > 90) & (hue < 130) & (sat > 30)
    blue_ratio = float(np.mean(blue_mask))

    # Warm ratio (skin/indoor/food): hue < 20 or hue > 160, saturation > 40
    warm_mask = ((hue < 20) | (hue > 160)) & (sat > 40)
    warm_ratio = float(np.mean(warm_mask))

    # ---- Edge density (texture complexity) ----
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.mean(edges > 0))

    # ---- Sky detection ----
    sky_sat = float(np.mean(sky_region[:, :, 1]))
    is_sky = sky_brightness > 140 and sky_sat < 40

    # ---- Dark region detection ----
    dark_ratio = float(np.mean(gray < 40))

    # ---- Build scores from features ----
    scores: dict[str, float] = {}

    # Cave/tunnel: very dark overall OR large dark region + low edge density
    if dark_ratio > 0.5:
        scores["隧道/洞穴"] = min(dark_ratio + 0.2, 1.0)
    elif dark_ratio > 0.2 and avg_brightness < 80:
        scores["隧道/洞穴"] = 0.7
        scores["夜景/暗光"] = 0.5

    # Night scene: dark but some bright spots
    if avg_brightness < 50 and dark_ratio < 0.5:
        scores["夜景/暗光"] = 0.8
        scores["室内/房间"] = 0.4

    # Indoor: moderate brightness, low edge density, low sky, warm tones
    if 40 < avg_brightness < 120 and not is_sky and warm_ratio > 0.15:
        scores["室内/房间"] = warm_ratio + 0.3
        if warm_ratio > 0.3:
            scores["餐厅/吃饭"] = 0.6

    # Outdoor natural: sky + green + moderate-high brightness
    if is_sky and green_ratio > 0.15:
        scores["户外自然风光"] = 0.6 + green_ratio * 0.4
        if green_ratio > 0.3:
            scores["山景"] = 0.5 + green_ratio * 0.4

    # Water scene: blue dominant + high sky brightness + low edge density
    if blue_ratio > 0.2 and is_sky and edge_density < 0.15:
        scores["海边/水边"] = 0.5 + blue_ratio * 0.4

    # Mountain: high edge density + green + sky
    if edge_density > 0.15 and green_ratio > 0.2 and is_sky:
        scores["山景"] = 0.5 + edge_density

    # Urban: low green, low blue, moderate edge, gray-ish
    avg_sat = float(np.mean(center_region[:, :, 1]))
    if avg_sat < 25 and is_sky and green_ratio < 0.1 and blue_ratio < 0.1:
        scores["城市街景"] = 0.6

    # Selfie/close-up: high warm ratio in center + cool edges (bokeh-like)
    center_warm = float(np.mean(warm_mask[h//3:2*h//3, w//3:2*w//3]))
    edge_warm = float(np.mean(np.concatenate([
        warm_mask[0:h//4, :].flatten(), warm_mask[-h//4:, :].flatten(),
        warm_mask[:, 0:w//4].flatten(), warm_mask[:, -w//4:].flatten(),
    ])))
    if center_warm > 0.4 and edge_warm < 0.2:
        scores["特写/自拍"] = 0.6

    # Driving/in-car: dark borders + bright center + low edge density
    border = np.concatenate([
        gray[0:h//8, :].flatten(),
        gray[-h//8:, :].flatten(),
        gray[:, 0:w//8].flatten(),
        gray[:, -w//8:].flatten(),
    ])
    border_dark = float(np.mean(border < 50))
    center_bright = float(np.mean(gray[h//3:2*h//3, w//3:2*w//3] > 100))
    if border_dark > 0.6 and center_bright > 0.4:
        scores["车内/驾驶"] = 0.7

    # ---- Texture analysis (Sobel edge density) ----
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    texture = np.sqrt(sobel_x**2 + sobel_y**2)
    texture_mean = float(np.mean(texture))
    texture_var = float(np.var(texture))

    # ---- Broader classification tiers (before fallback) ----

    # Natural outdoor: green or sky, moderate to bright
    if not scores and avg_brightness > 100:
        if green_ratio > 0.1:
            scores["户外自然风光"] = 0.4 + green_ratio * 0.4
        elif is_sky:
            scores["户外自然风光"] = 0.45
        elif blue_ratio > 0.1:
            scores["海边/水边"] = 0.4 + blue_ratio * 0.3

    # Water scene: smooth (low texture) + blue + bright
    if not scores and blue_ratio > 0.1 and texture_var < 100 and avg_brightness > 80:
        scores["海边/水边"] = 0.5 + blue_ratio * 0.3

    # Indoor: moderate brightness, moderate texture, not sky
    if not scores and 50 < avg_brightness < 130 and not is_sky:
        if warm_ratio > 0.1:
            scores["室内/房间"] = 0.4 + warm_ratio * 0.3
        elif edge_density < 0.1:
            scores["室内/房间"] = 0.45
        else:
            scores["室内/房间"] = 0.35

    # Mountain/forest: high texture + green
    if not scores and green_ratio > 0.1 and texture_var > 200:
        scores["山景"] = 0.45 + green_ratio * 0.3

    # Urban: moderate texture, low saturation, not green/blue
    if not scores and edge_density > 0.05 and avg_sat < 30 and green_ratio < 0.1 and blue_ratio < 0.1:
        scores["城市街景"] = 0.4

    # Night/dark
    if not scores and avg_brightness < 50:
        if dark_ratio > 0.3:
            scores["夜景/暗光"] = 0.5 + dark_ratio * 0.3
        else:
            scores["夜景/暗光"] = 0.4

    # Final fallback: if we know nothing, use brightness + texture to guess
    if not scores:
        if avg_brightness > 120 and texture_var > 100:
            scores["户外自然风光"] = 0.35
        elif avg_brightness > 120:
            scores["户外自然风光"] = 0.3
        elif avg_brightness > 60:
            scores["室内/房间"] = 0.3
        else:
            scores["夜景/暗光"] = 0.3

    # Build result sorted by confidence
    tags = sorted(
        [SceneTag(label=k, confidence=round(min(v, 1.0), 3), source="heuristic")
         for k, v in scores.items()],
        key=lambda t: t.confidence,
        reverse=True,
    )

    if not tags:
        tags = [SceneTag(label="普通场景", confidence=0.5, source="heuristic")]

    return tags[:3]
