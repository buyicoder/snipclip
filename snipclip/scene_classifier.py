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
    """Scene classification based on color and brightness analysis."""
    img = imread(image_path)
    if img is None:
        return [SceneTag(label="未知场景", confidence=0.0, source="heuristic")]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = img.shape[:2]

    # Split into regions for more nuanced analysis
    center = hsv[h//4:3*h//4, w//4:3*w//4]  # center region

    # Average color metrics
    avg_brightness = float(np.mean(center[:, :, 2]))
    avg_saturation = float(np.mean(center[:, :, 1]))
    avg_hue = float(np.mean(center[:, :, 0]))

    scores: dict[str, float] = {}

    # Darkness → night/indoor/tunnel
    if avg_brightness < 60:
        scores["夜景/暗光"] = 0.8
        scores["隧道/洞穴"] = 0.5
    elif avg_brightness < 100:
        scores["室内/房间"] = 0.7
        scores["隧道/洞穴"] = 0.6
        scores["餐厅/吃饭"] = 0.4
    else:
        scores["户外自然风光"] = 0.6

        # Hue analysis
        if 35 < avg_hue < 85:
            scores["户外自然风光"] = 0.85  # green
            scores["山景"] = 0.7
        elif 90 < avg_hue < 130:
            scores["海边/水边"] = 0.8  # blue
        elif avg_saturation < 30:
            scores["城市街景"] = 0.6
            scores["车内/驾驶"] = 0.4

    # Warm tones → restaurant/indoor
    if avg_hue < 20 and avg_saturation > 50:
        scores["餐厅/吃饭"] = max(scores.get("餐厅/吃饭", 0), 0.7)
        scores["室内/房间"] = max(scores.get("室内/房间", 0), 0.6)

    # High saturation green → nature/outdoor
    if avg_hue > 40 and avg_hue < 80 and avg_saturation > 60:
        scores["山景"] = 0.8
        scores["户外自然风光"] = 0.9

    # Very bright + blue → water/sky
    if avg_brightness > 150 and avg_hue > 100:
        scores["海边/水边"] = 0.7

    # Portrait aspect shots → selfie/portrait
    if h > w * 1.3:
        scores["特写/自拍"] = 0.7

    # Build result
    tags = sorted(
        [SceneTag(label=k, confidence=round(v, 3), source="heuristic") for k, v in scores.items()],
        key=lambda t: t.confidence,
        reverse=True,
    )

    if not tags:
        tags = [SceneTag(label="普通场景", confidence=0.5, source="heuristic")]

    return tags[:3]  # top 3
