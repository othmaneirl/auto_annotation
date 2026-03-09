"""
YOLO detection wrapper for auto-annotation pipeline.

Supports pretrained COCO models and custom fine-tuned models via the
ultralytics library. Returns detections in a standardized format.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


Detection = Dict[str, Any]


class Detector:
    """Wraps an ultralytics YOLO model for inference.

    Parameters
    ----------
    model_path:
        Path to a ``.pt`` weights file or a pretrained model name such as
        ``"yolov8n.pt"``.
    conf_threshold:
        Default confidence threshold used when ``detect()`` is called without
        an explicit value.
    """

    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.25) -> None:
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self._model = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_model(self, model_path: Optional[str] = None) -> None:
        """Load (or reload) the YOLO model.

        Parameters
        ----------
        model_path:
            If provided, overrides ``self.model_path`` for this load.
        """
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "ultralytics is required. Install it with: pip install ultralytics"
            ) from exc

        path = model_path or self.model_path
        logger.info("Loading YOLO model from: %s", path)
        self._model = YOLO(path)
        self.model_path = path
        logger.info("Model loaded successfully.")

    def detect(
        self,
        image_path: str,
        conf_threshold: Optional[float] = None,
    ) -> List[Detection]:
        """Run inference on a single image.

        Parameters
        ----------
        image_path:
            Absolute or relative path to the image file.
        conf_threshold:
            Confidence threshold to use for this call. Falls back to
            ``self.conf_threshold`` when *None*.

        Returns
        -------
        List[Detection]
            A list of dicts with keys ``bbox`` (``[x1, y1, x2, y2]`` in
            pixel coordinates), ``class`` (class name string), and
            ``confidence`` (float in ``[0, 1]``).
        """
        if self._model is None:
            self.load_model()

        threshold = conf_threshold if conf_threshold is not None else self.conf_threshold
        image_path = str(image_path)

        try:
            results = self._model(image_path, conf=threshold, verbose=False)
        except Exception as exc:
            logger.error("Detection failed for %s: %s", image_path, exc)
            return []

        detections: List[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].tolist()
                conf = float(boxes.conf[i])
                cls_id = int(boxes.cls[i])
                cls_name = (
                    result.names[cls_id]
                    if result.names and cls_id in result.names
                    else str(cls_id)
                )
                detections.append(
                    {
                        "bbox": [float(v) for v in xyxy],
                        "class": cls_name,
                        "confidence": conf,
                    }
                )
        return detections

    def detect_batch(
        self,
        image_paths: List[str],
        conf_threshold: Optional[float] = None,
    ) -> Dict[str, List[Detection]]:
        """Run inference on a list of images.

        Parameters
        ----------
        image_paths:
            List of image file paths.
        conf_threshold:
            Confidence threshold passed to each ``detect()`` call.

        Returns
        -------
        Dict[str, List[Detection]]
            Mapping of image path → list of detections.
        """
        results: Dict[str, List[Detection]] = {}
        for path in image_paths:
            results[path] = self.detect(path, conf_threshold=conf_threshold)
        return results

    @property
    def is_loaded(self) -> bool:
        """Return *True* if the model is currently loaded."""
        return self._model is not None

    @property
    def class_names(self) -> Dict[int, str]:
        """Return the class-id → name mapping of the loaded model."""
        if self._model is None:
            return {}
        return dict(self._model.names) if hasattr(self._model, "names") else {}
