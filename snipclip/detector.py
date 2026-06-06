"""Object and face detection using OpenCV.

Face detection via Haar Cascade (built-in, no extra files needed).
Object detection via ONNX YOLOv8-nano (optional, auto-downloads model).
"""

from pathlib import Path
from typing import List, NamedTuple, Optional
import os

import cv2
import numpy as np
from snipclip._cv import imread


# COCO class names for YOLO
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


class DetectedObject(NamedTuple):
    """A detected object in an image."""
    label: str
    confidence: float
    bbox: tuple    # (x, y, w, h)


class DetectionResult(NamedTuple):
    """Complete detection result for an image."""
    objects: List[DetectedObject]
    faces: int
    people_count: int


# Global cache for YOLO model
_yolo_model: Optional[cv2.dnn.Net] = None
_face_cascade: Optional[cv2.CascadeClassifier] = None


def _get_face_cascade() -> cv2.CascadeClassifier:
    """Load OpenCV Haar cascade for face detection."""
    global _face_cascade
    if _face_cascade is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(cascade_path)
    return _face_cascade


def _get_yolo_model() -> Optional[cv2.dnn.Net]:
    """Load YOLOv8-nano ONNX model. Auto-downloads if needed."""
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model

    model_dir = Path.home() / ".snipclip" / "models"
    model_path = model_dir / "yolov8n.onnx"

    if not model_path.exists():
        return None  # Silently degrade — no YOLO

    _yolo_model = cv2.dnn.readNetFromONNX(str(model_path))
    return _yolo_model


def download_yolo_model() -> Path:
    """Download YOLOv8-nano ONNX model. Returns path to downloaded file."""
    import urllib.request

    model_dir = Path.home() / ".snipclip" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "yolov8n.onnx"

    if not model_path.exists():
        url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.onnx"
        print(f"Downloading YOLOv8-nano ONNX model (~6MB)...")
        urllib.request.urlretrieve(url, str(model_path))
        print(f"Model saved to {model_path}")

    return model_path


def detect(image_path: Path) -> DetectionResult:
    """Detect faces and objects in an image.

    Face detection always works (OpenCV built-in).
    Object detection works only if YOLO model is downloaded.

    Args:
        image_path: Path to image file.

    Returns:
        DetectionResult with detected objects and face count.
    """
    img = imread(image_path)
    if img is None:
        return DetectionResult(objects=[], faces=0, people_count=0)

    # Face detection
    cascade = _get_face_cascade()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 3, minSize=(30, 30))

    # YOLO object detection (if available)
    objects: List[DetectedObject] = []
    yolo = _get_yolo_model()
    if yolo is not None:
        objects = _run_yolo(yolo, img)

    # Count people (from YOLO if available, else from faces)
    people_count = sum(1 for o in objects if o.label == "person")
    if not people_count:
        people_count = len(faces)

    return DetectionResult(
        objects=objects,
        faces=len(faces),
        people_count=people_count,
    )


def _run_yolo(model: cv2.dnn.Net, img: np.ndarray) -> List[DetectedObject]:
    """Run YOLO inference on an image."""
    h, w = img.shape[:2]

    # Preprocess
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (640, 640), swapRB=True, crop=False)
    model.setInput(blob)

    # Inference
    outputs = model.forward()

    # Parse outputs (YOLOv8 format: [1, 84, 8400])
    outputs = np.transpose(outputs[0])  # [8400, 84]

    objects: List[DetectedObject] = []
    for detection in outputs:
        scores = detection[4:]
        class_id = int(np.argmax(scores))
        confidence = float(scores[class_id])

        if confidence < 0.4:
            continue

        # Bbox
        cx, cy, bw, bh = detection[:4]
        x = int((cx - bw / 2) * w / 640)
        y = int((cy - bh / 2) * h / 640)
        bw = int(bw * w / 640)
        bh = int(bh * h / 640)

        if class_id < len(COCO_CLASSES):
            objects.append(DetectedObject(
                label=COCO_CLASSES[class_id],
                confidence=round(confidence, 3),
                bbox=(max(0, x), max(0, y), min(w, bw), min(h, bh)),
            ))

    return objects
