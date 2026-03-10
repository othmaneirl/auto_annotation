"""
Uncertainty sampling strategies for the active learning pipeline.

Images whose detections have the lowest average confidence are considered
the most uncertain and should be prioritised for human review.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Type alias
DetectionMap = Dict[str, List[Dict]]  # image_path -> list of detection dicts


def image_uncertainty_score(detections: List[Dict]) -> float:
    """Compute an uncertainty score for a single image.

    Images **with** detections are scored by their average confidence
    (lower confidence → more uncertain → reviewed first).

    Images with **no detections** receive a score of ``2.0`` so they are
    placed *after* images that have some predictions the user can correct.
    Reviewing images where the model already drew boxes is far more
    productive than starting annotation from scratch.

    Parameters
    ----------
    detections:
        List of detection dicts as returned by :class:`~detector.Detector`.
        Each dict must contain a ``"confidence"`` key with a float value.

    Returns
    -------
    float
        Average confidence in ``[0, 1]`` for images with detections, or
        ``2.0`` for images with none.
    """
    if not detections:
        return 2.0  # push "nothing detected" to the back
    return sum(d["confidence"] for d in detections) / len(detections)


def select_uncertain_images(
    results: DetectionMap,
    n: int = 100,
    reviewed: Optional[List[str]] = None,
) -> List[str]:
    """Select the *n* most uncertain image paths from a detection result map.

    Uncertainty is defined as the average confidence of detections (lower
    confidence == higher uncertainty).  Images with zero detections are
    treated as the most uncertain.

    Parameters
    ----------
    results:
        Mapping of ``image_path → list[detection_dict]``.
    n:
        Number of images to select.
    reviewed:
        Optional list of already-reviewed image paths to exclude from
        selection.

    Returns
    -------
    List[str]
        Up to *n* image paths sorted by ascending average confidence (most
        uncertain first).
    """
    reviewed_set = set(reviewed or [])
    candidates = {
        path: detections
        for path, detections in results.items()
        if path not in reviewed_set
    }

    scored = [
        (path, image_uncertainty_score(dets))
        for path, dets in candidates.items()
    ]
    # Sort ascending: lowest confidence (most uncertain) first
    scored.sort(key=lambda x: x[1])

    selected = [path for path, _ in scored[:n]]
    logger.info(
        "Selected %d uncertain images (from %d candidates, %d reviewed)",
        len(selected),
        len(candidates),
        len(reviewed_set),
    )
    return selected


def auto_annotate_batch(
    image_paths: List[str],
    detector,
    annotator,
    batch_size: int = 1000,
    conf_threshold: float = 0.25,
) -> DetectionMap:
    """Run the detector on a batch of images and save results automatically.

    This function is used in the *auto-annotate* phase of the active learning
    loop: it processes up to ``batch_size`` images, saves YOLO labels for
    each, and returns the detection map (used subsequently for uncertainty
    sampling).

    Parameters
    ----------
    image_paths:
        Candidate image paths to process.
    detector:
        A :class:`~detector.Detector` instance (model must already be loaded).
    annotator:
        An :class:`~annotator.AnnotationManager` instance used to persist
        labels and manage class mappings.
    batch_size:
        Maximum number of images to process in this call.
    conf_threshold:
        Confidence threshold passed to the detector.

    Returns
    -------
    DetectionMap
        Mapping of ``image_path → list[detection_dict]`` for every image that
        was processed.
    """
    from PIL import Image  # type: ignore  # lazy import

    subset = image_paths[:batch_size]
    results: DetectionMap = {}

    for idx, path in enumerate(subset, start=1):
        try:
            img = Image.open(path)
            img_w, img_h = img.size
        except Exception as exc:
            logger.warning("Cannot open image %s: %s", path, exc)
            continue

        detections = detector.detect(path, conf_threshold=conf_threshold)

        # Register any new classes discovered by the model
        for det in detections:
            if annotator.get_class_id(det["class"]) is None:
                annotator.add_class(det["class"])

        annotator.save_annotations(path, detections, img_w, img_h)
        results[path] = detections

        if idx % 10 == 0 or idx == len(subset):
            logger.info("Auto-annotated %d / %d images …", idx, len(subset))

    logger.info("Auto-annotation complete: %d images processed.", len(results))
    return results


class ActiveLearningTracker:
    """Track rounds, reviewed images, and model versions across sessions.

    Attributes
    ----------
    round_number:
        Current active learning round (0-indexed).
    reviewed_images:
        Set of image paths that the user has already reviewed.
    model_versions:
        List of model file paths saved after each training round.
    """

    def __init__(self) -> None:
        self.round_number: int = 0
        self.reviewed_images: List[str] = []
        self.model_versions: List[str] = []

    def mark_reviewed(self, image_paths: List[str]) -> None:
        """Record that *image_paths* have been reviewed by the user."""
        existing = set(self.reviewed_images)
        for p in image_paths:
            if p not in existing:
                self.reviewed_images.append(p)
                existing.add(p)

    def add_model_version(self, model_path: str) -> None:
        """Record a newly trained model version."""
        self.model_versions.append(model_path)
        logger.info("Registered model version: %s (round %d)", model_path, self.round_number)

    def advance_round(self) -> None:
        """Increment the round counter."""
        self.round_number += 1
        logger.info("Advanced to round %d", self.round_number)

    @property
    def latest_model(self) -> Optional[str]:
        """Return the path to the most recently trained model, or *None*."""
        return self.model_versions[-1] if self.model_versions else None

    def summary(self) -> Dict:
        """Return a summary dict for display in the UI."""
        return {
            "round": self.round_number,
            "reviewed_images": len(self.reviewed_images),
            "model_versions": len(self.model_versions),
            "latest_model": self.latest_model,
        }
