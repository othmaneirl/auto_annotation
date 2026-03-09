"""
Annotation management for the auto-annotation pipeline.

Handles reading and writing YOLO-format label files, class mappings,
and dataset configuration files.

YOLO label format (one box per line):
    <class_id> <x_center> <y_center> <width> <height>
All coordinates are normalized to ``[0, 1]`` relative to image dimensions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml  # type: ignore

logger = logging.getLogger(__name__)

# Type aliases
BboxXYXY = List[float]  # [x1, y1, x2, y2] in pixel coords
BboxYOLO = Tuple[float, float, float, float]  # (x_c, y_c, w, h) normalized


def _clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamp *value* to ``[min_val, max_val]``."""
    return max(min_val, min(max_val, value))


class AnnotationManager:
    """Manage YOLO-format annotation files and class mappings.

    Parameters
    ----------
    labels_dir:
        Root directory where ``.txt`` label files are written.
    classes_file:
        Path to the plain-text file listing one class name per line.
    """

    def __init__(self, labels_dir: str, classes_file: str) -> None:
        self.labels_dir = Path(labels_dir)
        self.classes_file = Path(classes_file)
        self._class_to_id: Dict[str, int] = {}
        self._id_to_class: Dict[int, str] = {}

        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self._load_classes()

    # ------------------------------------------------------------------
    # Class management
    # ------------------------------------------------------------------

    def _load_classes(self) -> None:
        """Load class names from ``self.classes_file`` (creates it if absent)."""
        if self.classes_file.exists():
            lines = self.classes_file.read_text(encoding="utf-8").splitlines()
            for idx, name in enumerate(lines):
                name = name.strip()
                if name:
                    self._class_to_id[name] = idx
                    self._id_to_class[idx] = name
        else:
            self.classes_file.parent.mkdir(parents=True, exist_ok=True)
            self.classes_file.touch()

    def _save_classes(self) -> None:
        """Persist the current class list to ``self.classes_file``."""
        names = [self._id_to_class[i] for i in sorted(self._id_to_class)]
        self.classes_file.write_text("\n".join(names) + "\n", encoding="utf-8")

    def add_class(self, name: str) -> int:
        """Add a new class if it doesn't already exist.

        Parameters
        ----------
        name:
            Human-readable class name.

        Returns
        -------
        int
            The integer ID assigned to this class.
        """
        name = name.strip()
        if name in self._class_to_id:
            return self._class_to_id[name]
        new_id = len(self._class_to_id)
        self._class_to_id[name] = new_id
        self._id_to_class[new_id] = name
        self._save_classes()
        logger.info("Added class '%s' with id %d", name, new_id)
        return new_id

    def get_class_id(self, name: str) -> Optional[int]:
        """Return the integer ID for a class name, or *None* if unknown."""
        return self._class_to_id.get(name.strip())

    def get_class_name(self, class_id: int) -> Optional[str]:
        """Return the class name for an integer ID, or *None* if unknown."""
        return self._id_to_class.get(class_id)

    @property
    def classes(self) -> List[str]:
        """Ordered list of class names (index == class id)."""
        return [self._id_to_class[i] for i in sorted(self._id_to_class)]

    @property
    def num_classes(self) -> int:
        """Number of known classes."""
        return len(self._class_to_id)

    # ------------------------------------------------------------------
    # Format conversions
    # ------------------------------------------------------------------

    @staticmethod
    def xyxy_to_yolo(
        bbox: BboxXYXY,
        img_width: int,
        img_height: int,
    ) -> BboxYOLO:
        """Convert absolute ``[x1, y1, x2, y2]`` to normalized YOLO format.

        Parameters
        ----------
        bbox:
            Bounding box as ``[x1, y1, x2, y2]`` in pixel coordinates.
        img_width:
            Image width in pixels.
        img_height:
            Image height in pixels.

        Returns
        -------
        tuple
            ``(x_center, y_center, width, height)`` all normalized to ``[0, 1]``.
        """
        x1, y1, x2, y2 = bbox
        x_c = ((x1 + x2) / 2) / img_width
        y_c = ((y1 + y2) / 2) / img_height
        w = (x2 - x1) / img_width
        h = (y2 - y1) / img_height
        return (
            _clamp(x_c),
            _clamp(y_c),
            _clamp(w),
            _clamp(h),
        )

    @staticmethod
    def yolo_to_xyxy(
        yolo_box: BboxYOLO,
        img_width: int,
        img_height: int,
    ) -> BboxXYXY:
        """Convert normalized YOLO format to absolute ``[x1, y1, x2, y2]``.

        Parameters
        ----------
        yolo_box:
            ``(x_center, y_center, width, height)`` normalized to ``[0, 1]``.
        img_width:
            Image width in pixels.
        img_height:
            Image height in pixels.

        Returns
        -------
        list
            ``[x1, y1, x2, y2]`` in pixel coordinates.
        """
        x_c, y_c, w, h = yolo_box
        x1 = (x_c - w / 2) * img_width
        y1 = (y_c - h / 2) * img_height
        x2 = (x_c + w / 2) * img_width
        y2 = (y_c + h / 2) * img_height
        return [x1, y1, x2, y2]

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def _label_path(self, image_path: str) -> Path:
        """Return the label ``.txt`` path for an image."""
        stem = Path(image_path).stem
        return self.labels_dir / f"{stem}.txt"

    def save_annotations(
        self,
        image_path: str,
        annotations: List[Dict],
        img_width: int,
        img_height: int,
    ) -> None:
        """Save a list of annotation dicts to a YOLO-format ``.txt`` file.

        Each annotation dict must contain:
        - ``"bbox"``: ``[x1, y1, x2, y2]`` in pixel coords.
        - ``"class"``: class name string.

        Parameters
        ----------
        image_path:
            Path to the source image (used to derive the label filename).
        annotations:
            List of annotation dictionaries.
        img_width:
            Image width in pixels.
        img_height:
            Image height in pixels.
        """
        label_path = self._label_path(image_path)
        lines: List[str] = []
        for ann in annotations:
            class_name = ann["class"]
            class_id = self.get_class_id(class_name)
            if class_id is None:
                class_id = self.add_class(class_name)
            yolo_box = self.xyxy_to_yolo(ann["bbox"], img_width, img_height)
            x_c, y_c, w, h = yolo_box
            lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        logger.debug("Saved %d annotations to %s", len(lines), label_path)

    def load_annotations(
        self,
        image_path: str,
        img_width: int,
        img_height: int,
    ) -> List[Dict]:
        """Load annotations from a YOLO ``.txt`` file for the given image.

        Parameters
        ----------
        image_path:
            Path to the source image.
        img_width:
            Image width in pixels (used for coordinate conversion).
        img_height:
            Image height in pixels (used for coordinate conversion).

        Returns
        -------
        List[Dict]
            List of dicts with ``"bbox"`` (pixel coords) and ``"class"``
            (class name string).
        """
        label_path = self._label_path(image_path)
        if not label_path.exists():
            return []

        annotations: List[Dict] = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                logger.warning("Skipping malformed label line: %s", line)
                continue
            class_id = int(parts[0])
            yolo_box: BboxYOLO = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
            bbox = self.yolo_to_xyxy(yolo_box, img_width, img_height)
            class_name = self.get_class_name(class_id) or str(class_id)
            annotations.append({"bbox": bbox, "class": class_name})
        return annotations

    def has_annotations(self, image_path: str) -> bool:
        """Return *True* if a non-empty label file exists for this image."""
        p = self._label_path(image_path)
        if not p.exists():
            return False
        return bool(p.read_text(encoding="utf-8").strip())

    # ------------------------------------------------------------------
    # Dataset YAML
    # ------------------------------------------------------------------

    def generate_dataset_yaml(
        self,
        yaml_path: str,
        train_images_dir: str,
        val_images_dir: str,
    ) -> str:
        """Write a ``dataset.yaml`` compatible with ultralytics YOLO training.

        Parameters
        ----------
        yaml_path:
            Output path for the YAML file.
        train_images_dir:
            Path to the directory containing training images.
        val_images_dir:
            Path to the directory containing validation images.

        Returns
        -------
        str
            The path to the written YAML file.
        """
        data = {
            "path": str(Path(yaml_path).parent.resolve()),
            "train": str(Path(train_images_dir).resolve()),
            "val": str(Path(val_images_dir).resolve()),
            "nc": self.num_classes,
            "names": self.classes,
        }
        out = Path(yaml_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            yaml.dump(data, fh, default_flow_style=False, allow_unicode=True)
        logger.info("Dataset YAML written to %s", out)
        return str(out)
