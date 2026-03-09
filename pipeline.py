"""
Core pipeline orchestration for the active learning auto-annotation system.

This module ties together detection, annotation, active learning, and
training into a coherent loop.

Active Learning Loop
--------------------
Round 0:
    1. Run pretrained YOLO on all images in ``data/unlabeled/``.
    2. Present detected boxes to the user via the Gradio UI.
    3. User corrects/assigns classes and saves annotations.
    4. Fine-tune the model on corrected data.

Round 1 +:
    5. Run the fine-tuned model on remaining unlabeled images.
    6. Auto-annotate up to ``batch_size_auto_annotate`` images.
    7. Pick ``batch_size_review`` least-confident images for user review.
    8. User corrects → retrain → repeat.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from active_learning import ActiveLearningTracker, auto_annotate_batch, select_uncertain_images
from annotator import AnnotationManager
from detector import Detector
from trainer import Trainer
from utils import collect_images, ensure_dirs, get_image_size, split_train_val

logger = logging.getLogger(__name__)


class PipelineConfig:
    """Configurable parameters for the pipeline with sensible defaults.

    Attributes
    ----------
    unlabeled_dir : str
        Directory containing unlabeled images.
    data_dir : str
        Root data directory (train/val sub-dirs are created here).
    runs_dir : str
        Root directory for training runs.
    base_model : str
        Pretrained model name or path used for round 0.
    conf_threshold : float
        Detection confidence threshold.
    batch_size_auto_annotate : int
        Number of images to auto-annotate per round.
    batch_size_review : int
        Number of uncertain images to present to the user per round.
    training_epochs_first : int
        Epochs for the first training round.
    training_epochs_subsequent : int
        Epochs for subsequent training rounds.
    image_size : int
        Training image size (pixels).
    val_ratio : float
        Fraction of annotated images to use for validation.
    """

    def __init__(
        self,
        unlabeled_dir: str = "data/unlabeled",
        data_dir: str = "data",
        runs_dir: str = "runs",
        base_model: str = "yolov8n.pt",
        conf_threshold: float = 0.25,
        batch_size_auto_annotate: int = 1000,
        batch_size_review: int = 100,
        training_epochs_first: int = 50,
        training_epochs_subsequent: int = 30,
        image_size: int = 640,
        val_ratio: float = 0.1,
    ) -> None:
        self.unlabeled_dir = unlabeled_dir
        self.data_dir = data_dir
        self.runs_dir = runs_dir
        self.base_model = base_model
        self.conf_threshold = conf_threshold
        self.batch_size_auto_annotate = batch_size_auto_annotate
        self.batch_size_review = batch_size_review
        self.training_epochs_first = training_epochs_first
        self.training_epochs_subsequent = training_epochs_subsequent
        self.image_size = image_size
        self.val_ratio = val_ratio

    def to_dict(self) -> Dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: Dict) -> "PipelineConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__init__.__code__.co_varnames})


class Pipeline:
    """Orchestrate the active learning auto-annotation loop.

    Parameters
    ----------
    config:
        Pipeline configuration.  A default :class:`PipelineConfig` is used
        when *None*.
    """

    STATE_FILE = "pipeline_state.json"

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self._setup_dirs()

        self.detector = Detector(
            model_path=self.config.base_model,
            conf_threshold=self.config.conf_threshold,
        )
        self.annotator = AnnotationManager(
            labels_dir=str(Path(self.config.data_dir) / "labels" / "train"),
            classes_file=str(Path(self.config.data_dir) / "classes.txt"),
        )
        self.trainer = Trainer(project_dir=self.config.runs_dir)
        self.tracker = ActiveLearningTracker()

        # Cache of detection results for the current batch
        self._current_detections: Dict[str, List[Dict]] = {}
        # Images currently queued for user review
        self._review_queue: List[str] = []

        self._load_state()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _setup_dirs(self) -> None:
        cfg = self.config
        ensure_dirs(
            cfg.unlabeled_dir,
            str(Path(cfg.data_dir) / "images" / "train"),
            str(Path(cfg.data_dir) / "images" / "val"),
            str(Path(cfg.data_dir) / "labels" / "train"),
            str(Path(cfg.data_dir) / "labels" / "val"),
            cfg.runs_dir,
        )

    # ------------------------------------------------------------------
    # State persistence (resumability)
    # ------------------------------------------------------------------

    def _state_path(self) -> Path:
        return Path(self.config.data_dir) / self.STATE_FILE

    def _save_state(self) -> None:
        state = {
            "round_number": self.tracker.round_number,
            "reviewed_images": self.tracker.reviewed_images,
            "model_versions": self.tracker.model_versions,
            "review_queue": self._review_queue,
        }
        self._state_path().write_text(json.dumps(state, indent=2), encoding="utf-8")
        logger.debug("Pipeline state saved.")

    def _load_state(self) -> None:
        p = self._state_path()
        if not p.exists():
            return
        try:
            state = json.loads(p.read_text(encoding="utf-8"))
            self.tracker.round_number = state.get("round_number", 0)
            self.tracker.reviewed_images = state.get("reviewed_images", [])
            self.tracker.model_versions = state.get("model_versions", [])
            self._review_queue = state.get("review_queue", [])
            logger.info("Pipeline state restored from %s", p)
        except Exception as exc:
            logger.warning("Could not load pipeline state: %s", exc)

    # ------------------------------------------------------------------
    # Round 0: initial detection
    # ------------------------------------------------------------------

    def run_initial_detection(self) -> List[str]:
        """Run the pretrained model on all unlabeled images.

        Returns
        -------
        List[str]
            Paths of images queued for user review (the full unlabeled set
            for round 0, or up to ``batch_size_review`` for subsequent rounds).
        """
        images = collect_images(self.config.unlabeled_dir)
        if not images:
            logger.warning("No images found in %s", self.config.unlabeled_dir)
            return []

        if not self.detector.is_loaded:
            self.detector.load_model()

        if self.tracker.round_number == 0:
            # Detect on ALL images for the very first round
            logger.info("Round 0: detecting on %d images …", len(images))
            for path in images:
                dets = self.detector.detect(path, conf_threshold=self.config.conf_threshold)
                self._current_detections[path] = dets

            # For round 0 pick the most uncertain ones up to batch_size_review
            uncertain = select_uncertain_images(
                self._current_detections,
                n=self.config.batch_size_review,
                reviewed=self.tracker.reviewed_images,
            )
            self._review_queue = uncertain
        else:
            # Subsequent rounds: auto-annotate a large batch first, then pick uncertain
            self._run_auto_annotation_round(images)

        self._save_state()
        return list(self._review_queue)

    def _run_auto_annotation_round(self, all_images: List[str]) -> None:
        """Auto-annotate a batch and select the most uncertain for review."""
        unlabeled = [p for p in all_images if p not in self.tracker.reviewed_images]
        logger.info(
            "Round %d: auto-annotating up to %d images …",
            self.tracker.round_number,
            self.config.batch_size_auto_annotate,
        )
        detection_map = auto_annotate_batch(
            unlabeled,
            detector=self.detector,
            annotator=self.annotator,
            batch_size=self.config.batch_size_auto_annotate,
            conf_threshold=self.config.conf_threshold,
        )
        self._current_detections.update(detection_map)

        uncertain = select_uncertain_images(
            detection_map,
            n=self.config.batch_size_review,
            reviewed=self.tracker.reviewed_images,
        )
        self._review_queue = uncertain

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _prepare_training_data(self) -> Optional[str]:
        """Copy reviewed images/labels into train/val splits and write YAML.

        Returns
        -------
        str or None
            Path to the dataset YAML, or *None* if there are no reviewed
            images yet.
        """
        reviewed = self.tracker.reviewed_images
        if not reviewed:
            logger.warning("No reviewed images available for training.")
            return None

        train_imgs, val_imgs = split_train_val(reviewed, val_ratio=self.config.val_ratio)

        train_img_dir = Path(self.config.data_dir) / "images" / "train"
        val_img_dir = Path(self.config.data_dir) / "images" / "val"
        train_lbl_dir = Path(self.config.data_dir) / "labels" / "train"
        val_lbl_dir = Path(self.config.data_dir) / "labels" / "val"

        for split_images, img_dir, lbl_dir in [
            (train_imgs, train_img_dir, train_lbl_dir),
            (val_imgs, val_img_dir, val_lbl_dir),
        ]:
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)
            for src in split_images:
                src_path = Path(src)
                dst_img = img_dir / src_path.name
                if not dst_img.exists():
                    shutil.copy2(src, dst_img)
                # Copy corresponding label
                src_lbl = Path(self.annotator.labels_dir) / f"{src_path.stem}.txt"
                dst_lbl = lbl_dir / f"{src_path.stem}.txt"
                if src_lbl.exists() and not dst_lbl.exists():
                    shutil.copy2(src_lbl, dst_lbl)

        yaml_path = str(Path(self.config.data_dir) / "dataset.yaml")
        self.annotator.generate_dataset_yaml(
            yaml_path,
            train_images_dir=str(train_img_dir),
            val_images_dir=str(val_img_dir),
        )
        return yaml_path

    def train_model(
        self,
        epochs: Optional[int] = None,
        device: str = "",
    ) -> Optional[str]:
        """Fine-tune the model on all reviewed & corrected annotations.

        Parameters
        ----------
        epochs:
            Override the default epoch count.
        device:
            Device string (e.g. ``"0"`` for GPU 0, ``"cpu"``).

        Returns
        -------
        str or None
            Path to the best checkpoint, or *None* if training could not
            be started.
        """
        yaml_path = self._prepare_training_data()
        if yaml_path is None:
            return None

        base_model = self.tracker.latest_model or self.config.base_model
        round_epochs = epochs or (
            self.config.training_epochs_first
            if self.tracker.round_number == 0
            else self.config.training_epochs_subsequent
        )

        best_pt = self.trainer.train(
            data_yaml=yaml_path,
            base_model=base_model,
            round_number=self.tracker.round_number,
            epochs=round_epochs,
            imgsz=self.config.image_size,
            device=device,
        )

        self.tracker.add_model_version(best_pt)

        # Reload the detector with the newly trained model
        self.detector.load_model(best_pt)
        self.tracker.advance_round()
        self._save_state()
        return best_pt

    # ------------------------------------------------------------------
    # Annotation helpers (called from the UI)
    # ------------------------------------------------------------------

    def save_user_annotations(
        self,
        image_path: str,
        annotations: List[Dict],
    ) -> None:
        """Persist user-corrected annotations for a single image.

        Parameters
        ----------
        image_path:
            Path to the image being annotated.
        annotations:
            List of dicts with ``"bbox"`` (pixel coords) and ``"class"``.
        """
        img_w, img_h = get_image_size(image_path)
        self.annotator.save_annotations(image_path, annotations, img_w, img_h)

    def mark_reviewed(self, image_paths: List[str]) -> None:
        """Mark images as reviewed and persist state."""
        self.tracker.mark_reviewed(image_paths)
        self._save_state()

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def review_queue(self) -> List[str]:
        """Images currently queued for human review."""
        return list(self._review_queue)

    @property
    def current_detections(self) -> Dict[str, List[Dict]]:
        """Detection results for images in the current review batch."""
        return dict(self._current_detections)

    def get_detections_for(self, image_path: str) -> List[Dict]:
        """Return cached detections for *image_path*, or run detection live."""
        if image_path in self._current_detections:
            return self._current_detections[image_path]
        dets = self.detector.detect(image_path, conf_threshold=self.config.conf_threshold)
        self._current_detections[image_path] = dets
        return dets

    def summary(self) -> Dict:
        """Return a high-level summary of pipeline state for the UI."""
        return {
            **self.tracker.summary(),
            "review_queue_size": len(self._review_queue),
            "unlabeled_images": len(collect_images(self.config.unlabeled_dir)),
            "num_classes": self.annotator.num_classes,
            "classes": self.annotator.classes,
        }
