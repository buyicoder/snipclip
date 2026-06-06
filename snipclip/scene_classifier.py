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

_clip_model: Optional = None
_clip_tokenizer: Optional = None


def _load_clip():
    """Try to load ONNX CLIP model. Returns (session, tokenizer) or (None, None)."""
    global _clip_model, _clip_tokenizer

    if _clip_model is not None:
        return _clip_model, _clip_tokenizer

    try:
        import onnxruntime as ort

        model_dir = Path.home() / ".snipclip" / "models" / "clip-vit-b32"
        model_path = model_dir / "model.onnx"

        if not model_path.exists():
            return None, None

        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

        # Load tokenizer (simple BPE) if available
        tokenizer_path = model_dir / "tokenizer.json"
        tokenizer = None
        if tokenizer_path.exists():
            try:
                from transformers import CLIPTokenizerFast
                tokenizer = CLIPTokenizerFast.from_pretrained(str(model_dir))
            except ImportError:
                pass

        _clip_model = session
        _clip_tokenizer = tokenizer
        return session, tokenizer

    except Exception:
        return None, None


def download_clip_model() -> Path:
    """Download CLIP-ViT-B-32 ONNX model from HuggingFace.

    Requires `transformers` for the tokenizer.
    """
    model_dir = Path.home() / ".snipclip" / "models" / "clip-vit-b32"
    model_path = model_dir / "model.onnx"

    if model_path.exists():
        return model_dir

    print("CLIP model not found. To enable scene classification, download:")
    print("  https://huggingface.co/Xenova/clip-vit-base-patch32")
    print(f"  and place ONNX files in: {model_dir}")
    print()
    print("Scene classification will use color heuristic instead.")
    return model_dir


def _classify_clip(image_path: Path, labels: List[str]) -> Optional[List[SceneTag]]:
    """Classify using ONNX CLIP. Returns None if model not available."""
    session, tokenizer = _load_clip()
    if session is None:
        return None

    try:
        # Load and preprocess image
        from PIL import Image

        img = Image.open(str(image_path)).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = (img_array - np.array([0.48145466, 0.4578275, 0.40821073])) / np.array([0.26862954, 0.26130258, 0.27577711])
        img_array = img_array.transpose(2, 0, 1)[np.newaxis, :]  # [1, 3, 224, 224]

        # Run image encoder
        img_input = {session.get_inputs()[0].name: img_array}
        img_features = session.run(None, img_input)[0]
        img_features = img_features / np.linalg.norm(img_features, axis=1, keepdims=True)

        if tokenizer is None:
            return None

        # Tokenize text labels
        text_inputs = tokenizer(labels, padding=True, truncation=True, return_tensors="np")
        text_out = session.run(None, {
            session.get_inputs()[1].name: text_inputs["input_ids"],
            session.get_inputs()[2].name: text_inputs["attention_mask"],
        })
        text_features = text_out[0]
        text_features = text_features / np.linalg.norm(text_features, axis=1, keepdims=True)

        # Cosine similarity
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

    # Fallback for bright outdoor without specific features
    if not scores and avg_brightness > 120 and is_sky:
        scores["户外自然风光"] = 0.5

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
