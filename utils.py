"""
Utility helpers for the auto-annotation pipeline.

Covers image I/O, bounding-box drawing, and common format conversions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict

logger = logging.getLogger(__name__)

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# Color range for random box colors (min and max channel values)
_COLOR_MIN = 50
_COLOR_MAX = 220


def collect_images(directory: str) -> List[str]:
    """Recursively collect image file paths from a directory.

    Parameters
    ----------
    directory:
        Root directory to scan.

    Returns
    -------
    List[str]
        Sorted list of absolute image file paths.
    """
    root = Path(directory)
    if not root.exists():
        logger.warning("Directory does not exist: %s", directory)
        return []
    paths = [
        str(p.resolve())
        for p in root.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    paths.sort()
    logger.info("Found %d images in %s", len(paths), directory)
    return paths


def get_image_size(image_path: str) -> Tuple[int, int]:
    """Return ``(width, height)`` of an image without fully decoding it.

    Parameters
    ----------
    image_path:
        Path to the image file.

    Returns
    -------
    tuple
        ``(width, height)`` in pixels.
    """
    from PIL import Image  # type: ignore

    with Image.open(image_path) as img:
        return img.size  # (width, height)


def draw_boxes(
    image_path: str,
    detections: List[Dict],
    class_colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
) -> "PIL.Image.Image":  # type: ignore[name-defined]  # noqa: F821
    """Draw bounding boxes on an image and return a PIL Image.

    Parameters
    ----------
    image_path:
        Path to the source image.
    detections:
        List of detection dicts with ``"bbox"`` (``[x1,y1,x2,y2]``),
        ``"class"`` and optionally ``"confidence"``.
    class_colors:
        Optional mapping of class name → RGB tuple.  Random colors are used
        for unknown classes.

    Returns
    -------
    PIL.Image.Image
        Annotated image.
    """
    from PIL import Image, ImageDraw, ImageFont  # type: ignore
    import random

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    _color_cache: Dict[str, Tuple[int, int, int]] = dict(class_colors or {})

    def _color(cls: str) -> Tuple[int, int, int]:
        if cls not in _color_cache:
            rng = random.Random(hash(cls) & 0xFFFFFFFF)
            _color_cache[cls] = (
                rng.randint(_COLOR_MIN, _COLOR_MAX),
                rng.randint(_COLOR_MIN, _COLOR_MAX),
                rng.randint(_COLOR_MIN, _COLOR_MAX),
            )
        return _color_cache[cls]

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cls = det.get("class", "?")
        conf = det.get("confidence", None)
        color = _color(cls)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"{cls} {conf:.2f}" if conf is not None else cls
        # Simple text background
        text_bbox = draw.textbbox((x1, y1 - 12), label)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x1, y1 - 12), label, fill=(255, 255, 255))

    return img


def split_train_val(
    image_paths: List[str],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """Split image paths into train and validation sets.

    Parameters
    ----------
    image_paths:
        Full list of image paths.
    val_ratio:
        Fraction of images to use for validation.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    tuple
        ``(train_paths, val_paths)``
    """
    import random

    paths = list(image_paths)
    random.seed(seed)
    random.shuffle(paths)
    n_val = max(1, int(len(paths) * val_ratio))
    return paths[n_val:], paths[:n_val]


def ensure_dirs(*dirs: str) -> None:
    """Create directories (and parents) if they don't already exist."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
