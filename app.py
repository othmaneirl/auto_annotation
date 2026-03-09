"""
Main Gradio application for the active learning auto-annotation pipeline.

Run with:
    python app.py

The UI exposes four tabs:
    1. Setup & Import   – choose images, model, thresholds, start detection
    2. Review & Annotate – correct/accept detections one image at a time
    3. Training & Auto-Annotation – fine-tune and run auto-annotation
    4. Export – download annotations and the trained model
"""

from __future__ import annotations

import io
import json
import logging
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy pipeline import (avoids heavy dependencies at module load for help text)
# ---------------------------------------------------------------------------

_pipeline: Optional[object] = None  # will be set to a Pipeline instance


def _get_pipeline():
    global _pipeline
    return _pipeline


def _set_pipeline(p):
    global _pipeline
    _pipeline = p


# ---------------------------------------------------------------------------
# Helper: draw boxes on an image and return a PIL image
# ---------------------------------------------------------------------------

def _render_image(image_path: str, annotations: List[Dict]) -> Optional["PIL.Image.Image"]:  # type: ignore[name-defined]  # noqa: F821
    try:
        from utils import draw_boxes
        return draw_boxes(image_path, annotations)
    except Exception as exc:
        logger.warning("Could not render image %s: %s", image_path, exc)
        return None


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

class SessionState:
    """Mutable session state shared across Gradio callbacks."""

    def __init__(self) -> None:
        self.review_images: List[str] = []
        self.current_index: int = 0
        # annotations[image_path] = list of dicts {"bbox": [...], "class": str, "confidence": float}
        self.annotations: Dict[str, List[Dict]] = {}
        self.pipeline_running: bool = False

    def current_image(self) -> Optional[str]:
        if 0 <= self.current_index < len(self.review_images):
            return self.review_images[self.current_index]
        return None

    def current_annotations(self) -> List[Dict]:
        img = self.current_image()
        if img is None:
            return []
        return self.annotations.get(img, [])

    def set_current_annotations(self, anns: List[Dict]) -> None:
        img = self.current_image()
        if img is not None:
            self.annotations[img] = anns


_session = SessionState()


# ===========================================================================
# Tab 1 – Setup & Import
# ===========================================================================

def setup_pipeline(
    unlabeled_dir: str,
    model_name: str,
    conf_threshold: float,
) -> str:
    """Initialise the pipeline and run the initial detection pass."""
    from pipeline import Pipeline, PipelineConfig

    unlabeled_dir = unlabeled_dir.strip()
    if not unlabeled_dir or not Path(unlabeled_dir).exists():
        return f"❌ Directory not found: '{unlabeled_dir}'"

    config = PipelineConfig(
        unlabeled_dir=unlabeled_dir,
        base_model=model_name.strip(),
        conf_threshold=float(conf_threshold),
    )
    p = Pipeline(config=config)
    _set_pipeline(p)

    msg = [f"✅ Pipeline initialised (model: {model_name}, threshold: {conf_threshold})."]
    msg.append("⏳ Running initial detection …")
    yield "\n".join(msg)

    try:
        queue = p.run_initial_detection()
    except Exception as exc:
        logger.exception("Detection failed")
        yield f"❌ Detection failed: {exc}"
        return

    _session.review_images = queue
    _session.current_index = 0
    _session.annotations = {}

    for path in queue:
        dets = p.get_detections_for(path)
        _session.annotations[path] = dets

    msg.append(
        f"✅ Detection complete. {len(queue)} images queued for review. "
        "→ Go to the **Review & Annotate** tab."
    )
    yield "\n".join(msg)


# ===========================================================================
# Tab 2 – Review & Annotate
# ===========================================================================

def _annotation_table(anns: List[Dict]) -> str:
    """Convert annotation list to a readable markdown table."""
    if not anns:
        return "_No annotations for this image._"
    header = "| # | Class | Confidence | BBox (x1,y1,x2,y2) |\n|---|-------|-----------|--------------------|\n"
    rows = []
    for i, a in enumerate(anns):
        bbox = a.get("bbox", [])
        conf = a.get("confidence", "—")
        conf_str = f"{conf:.2f}" if isinstance(conf, float) else str(conf)
        bbox_str = ", ".join(f"{v:.1f}" for v in bbox) if bbox else "—"
        rows.append(f"| {i} | {a.get('class','?')} | {conf_str} | {bbox_str} |")
    return header + "\n".join(rows)


def review_load_current() -> Tuple:
    """Return (image, annotation_table, progress_text, annotation_json)."""
    img_path = _session.current_image()
    if img_path is None:
        return None, "_No images to review._", "0 / 0", "[]"

    anns = _session.current_annotations()
    rendered = _render_image(img_path, anns)
    table = _annotation_table(anns)
    progress = f"Image {_session.current_index + 1} / {len(_session.review_images)}"
    ann_json = json.dumps(anns, indent=2)
    return rendered, table, progress, ann_json


def review_next() -> Tuple:
    """Advance to the next image."""
    if _session.current_index < len(_session.review_images) - 1:
        _session.current_index += 1
    return review_load_current()


def review_prev() -> Tuple:
    """Go back to the previous image."""
    if _session.current_index > 0:
        _session.current_index -= 1
    return review_load_current()


def save_and_next(annotation_json: str) -> Tuple:
    """Parse the JSON, persist annotations, advance to next image."""
    p = _get_pipeline()
    img_path = _session.current_image()
    if img_path is None:
        return review_load_current()

    try:
        anns = json.loads(annotation_json)
        if not isinstance(anns, list):
            anns = []
    except json.JSONDecodeError:
        anns = _session.current_annotations()

    _session.set_current_annotations(anns)
    if p is not None:
        try:
            p.save_user_annotations(img_path, anns)
            p.mark_reviewed([img_path])
        except Exception as exc:
            logger.warning("Could not save annotations for %s: %s", img_path, exc)

    return review_next()


def delete_annotation(annotation_json: str, index: int) -> Tuple:
    """Remove a single annotation by index."""
    try:
        anns = json.loads(annotation_json)
        if not isinstance(anns, list):
            anns = []
    except json.JSONDecodeError:
        anns = _session.current_annotations()

    if 0 <= index < len(anns):
        anns.pop(index)
    _session.set_current_annotations(anns)
    return _render_image(_session.current_image(), anns), _annotation_table(anns), json.dumps(anns, indent=2)


def add_annotation(annotation_json: str, new_class: str, x1: float, y1: float, x2: float, y2: float) -> Tuple:
    """Append a manually entered bounding box."""
    try:
        anns = json.loads(annotation_json)
        if not isinstance(anns, list):
            anns = []
    except json.JSONDecodeError:
        anns = _session.current_annotations()

    new_ann = {"bbox": [x1, y1, x2, y2], "class": new_class.strip(), "confidence": 1.0}
    anns.append(new_ann)

    # Register class if pipeline is ready
    p = _get_pipeline()
    if p is not None and new_class.strip():
        try:
            p.annotator.add_class(new_class.strip())
        except Exception:
            pass

    _session.set_current_annotations(anns)
    return _render_image(_session.current_image(), anns), _annotation_table(anns), json.dumps(anns, indent=2)


def change_class(annotation_json: str, index: int, new_class: str) -> Tuple:
    """Change the class of an existing annotation."""
    try:
        anns = json.loads(annotation_json)
        if not isinstance(anns, list):
            anns = []
    except json.JSONDecodeError:
        anns = _session.current_annotations()

    if 0 <= index < len(anns):
        anns[index]["class"] = new_class.strip()

    p = _get_pipeline()
    if p is not None and new_class.strip():
        try:
            p.annotator.add_class(new_class.strip())
        except Exception:
            pass

    _session.set_current_annotations(anns)
    return _render_image(_session.current_image(), anns), _annotation_table(anns), json.dumps(anns, indent=2)


def get_class_list() -> List[str]:
    p = _get_pipeline()
    if p is None:
        return []
    return p.annotator.classes


# ===========================================================================
# Tab 3 – Training & Auto-Annotation
# ===========================================================================

def run_training(epochs_override: int, device: str) -> str:
    p = _get_pipeline()
    if p is None:
        return "❌ Pipeline not initialised. Go to **Setup & Import** first."

    epoch_val = int(epochs_override) if epochs_override > 0 else None
    device_val = device.strip()

    try:
        best_pt = p.train_model(epochs=epoch_val, device=device_val or "")
    except Exception as exc:
        logger.exception("Training failed")
        return f"❌ Training failed: {exc}"

    if best_pt is None:
        return "⚠️ No reviewed images yet. Annotate at least one image first."

    return f"✅ Training complete.\nBest model: `{best_pt}`\nCurrent round: {p.tracker.round_number}"


def run_auto_annotation() -> str:
    p = _get_pipeline()
    if p is None:
        return "❌ Pipeline not initialised."

    from utils import collect_images

    images = collect_images(p.config.unlabeled_dir)
    if not images:
        return "⚠️ No images found in the unlabeled directory."

    try:
        queue = p.run_initial_detection()
    except Exception as exc:
        logger.exception("Auto-annotation failed")
        return f"❌ Auto-annotation failed: {exc}"

    _session.review_images = queue
    _session.current_index = 0
    _session.annotations = {}
    for path in queue:
        dets = p.get_detections_for(path)
        _session.annotations[path] = dets

    summary = p.summary()
    return (
        f"✅ Auto-annotation complete.\n"
        f"Round: {summary['round']}\n"
        f"Images queued for review: {summary['review_queue_size']}\n"
        f"Total reviewed so far: {summary['reviewed_images']}\n"
        f"Classes: {', '.join(summary['classes']) or 'none yet'}"
    )


def pipeline_summary() -> str:
    p = _get_pipeline()
    if p is None:
        return "_Pipeline not initialised._"
    s = p.summary()
    lines = [
        f"**Round:** {s['round']}",
        f"**Reviewed images:** {s['reviewed_images']}",
        f"**Review queue:** {s['review_queue_size']}",
        f"**Unlabeled images:** {s['unlabeled_images']}",
        f"**Classes ({s['num_classes']}):** {', '.join(s['classes']) or 'none'}",
        f"**Latest model:** {s['latest_model'] or 'pretrained'}",
    ]
    return "\n".join(lines)


# ===========================================================================
# Tab 4 – Export
# ===========================================================================

def export_annotations() -> Optional[str]:
    p = _get_pipeline()
    if p is None:
        return None

    out_zip = Path("export_annotations.zip")
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        labels_dir = Path(p.annotator.labels_dir)
        for txt_file in labels_dir.glob("*.txt"):
            zf.write(txt_file, arcname=f"labels/{txt_file.name}")
        classes_file = p.annotator.classes_file
        if classes_file.exists():
            zf.write(classes_file, arcname="classes.txt")
        data_dir = Path(p.config.data_dir)
        yaml_path = data_dir / "dataset.yaml"
        if yaml_path.exists():
            zf.write(yaml_path, arcname="dataset.yaml")

    return str(out_zip)


def export_model() -> Optional[str]:
    p = _get_pipeline()
    if p is None or p.tracker.latest_model is None:
        return None
    return p.tracker.latest_model


# ===========================================================================
# Build the Gradio UI
# ===========================================================================

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Auto-Annotation Pipeline") as demo:
        gr.Markdown(
            """
# 🤖 Active Learning Auto-Annotation Pipeline

Annotate object-detection datasets using pretrained YOLO + human-in-the-loop refinement.
            """
        )

        with gr.Tab("⚙️ Setup & Import"):
            gr.Markdown("### 1. Configure the pipeline")
            with gr.Row():
                unlabeled_dir_input = gr.Textbox(
                    label="Unlabeled images directory",
                    placeholder="data/unlabeled",
                    value="data/unlabeled",
                )
                model_choice = gr.Dropdown(
                    label="Base YOLO model",
                    choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
                    value="yolov8n.pt",
                )
                conf_slider = gr.Slider(
                    label="Confidence threshold",
                    minimum=0.05,
                    maximum=0.95,
                    step=0.05,
                    value=0.25,
                )
            start_btn = gr.Button("🚀 Start Initial Detection", variant="primary")
            setup_log = gr.Textbox(label="Status", lines=6, interactive=False)

            start_btn.click(
                fn=setup_pipeline,
                inputs=[unlabeled_dir_input, model_choice, conf_slider],
                outputs=[setup_log],
            )

        with gr.Tab("🖊️ Review & Annotate"):
            gr.Markdown(
                "Review detected bounding boxes, correct classes, delete false positives, "
                "or add missing boxes."
            )
            with gr.Row():
                progress_text = gr.Textbox(label="Progress", interactive=False, scale=1)
                class_list_display = gr.Textbox(label="Known classes", interactive=False, scale=3)

            with gr.Row():
                annotated_image = gr.Image(label="Current image", type="pil", interactive=False)

            annotation_table_md = gr.Markdown("_No image loaded._")
            annotation_json = gr.Textbox(
                label="Annotations JSON (editable)",
                lines=12,
                placeholder='[{"bbox": [x1,y1,x2,y2], "class": "cat", "confidence": 0.9}]',
            )

            with gr.Row():
                prev_btn = gr.Button("⬅ Previous")
                save_next_btn = gr.Button("💾 Save & Next", variant="primary")
                next_btn = gr.Button("Next ➡")

            gr.Markdown("#### Modify annotations")
            with gr.Row():
                del_index = gr.Number(label="Delete annotation #", value=0, precision=0)
                del_btn = gr.Button("🗑 Delete")
            with gr.Row():
                chg_index = gr.Number(label="Change class of annotation #", value=0, precision=0)
                chg_class = gr.Textbox(label="New class name", placeholder="dog")
                chg_btn = gr.Button("✏️ Change Class")
            gr.Markdown("#### Add new annotation")
            with gr.Row():
                new_class = gr.Textbox(label="Class name", placeholder="cat")
                x1_in = gr.Number(label="x1", value=0)
                y1_in = gr.Number(label="y1", value=0)
                x2_in = gr.Number(label="x2", value=100)
                y2_in = gr.Number(label="y2", value=100)
            add_btn = gr.Button("➕ Add Annotation")

            # Wire navigation buttons
            prev_btn.click(
                fn=review_prev,
                outputs=[annotated_image, annotation_table_md, progress_text, annotation_json],
            )
            next_btn.click(
                fn=review_next,
                outputs=[annotated_image, annotation_table_md, progress_text, annotation_json],
            )
            save_next_btn.click(
                fn=save_and_next,
                inputs=[annotation_json],
                outputs=[annotated_image, annotation_table_md, progress_text, annotation_json],
            )
            del_btn.click(
                fn=delete_annotation,
                inputs=[annotation_json, del_index],
                outputs=[annotated_image, annotation_table_md, annotation_json],
            )
            chg_btn.click(
                fn=change_class,
                inputs=[annotation_json, chg_index, chg_class],
                outputs=[annotated_image, annotation_table_md, annotation_json],
            )
            add_btn.click(
                fn=add_annotation,
                inputs=[annotation_json, new_class, x1_in, y1_in, x2_in, y2_in],
                outputs=[annotated_image, annotation_table_md, annotation_json],
            )

            # Refresh class list on tab focus (approximated via a button)
            refresh_classes_btn = gr.Button("🔄 Refresh class list")
            refresh_classes_btn.click(
                fn=lambda: ", ".join(get_class_list()) or "none",
                outputs=[class_list_display],
            )

            # Load first image button
            load_btn = gr.Button("📂 Load current image")
            load_btn.click(
                fn=review_load_current,
                outputs=[annotated_image, annotation_table_md, progress_text, annotation_json],
            )

        with gr.Tab("🏋️ Training & Auto-Annotation"):
            gr.Markdown("Fine-tune the model on corrected data, then auto-annotate remaining images.")

            with gr.Row():
                epochs_input = gr.Number(label="Epochs (0 = auto)", value=0, precision=0)
                device_input = gr.Textbox(label="Device (blank = auto)", placeholder="cpu")

            train_btn = gr.Button("🚂 Fine-tune Model", variant="primary")
            train_output = gr.Textbox(label="Training log", lines=8, interactive=False)

            auto_btn = gr.Button("🤖 Run Auto-Annotation & Select Uncertain Images")
            auto_output = gr.Textbox(label="Auto-annotation log", lines=8, interactive=False)

            summary_btn = gr.Button("📊 Refresh Pipeline Summary")
            summary_output = gr.Markdown("_Click 'Refresh Pipeline Summary' to update._")

            train_btn.click(
                fn=run_training,
                inputs=[epochs_input, device_input],
                outputs=[train_output],
            )
            auto_btn.click(
                fn=run_auto_annotation,
                outputs=[auto_output],
            )
            summary_btn.click(
                fn=pipeline_summary,
                outputs=[summary_output],
            )

        with gr.Tab("📦 Export"):
            gr.Markdown("Download your annotations and the trained model.")
            export_ann_btn = gr.Button("📥 Export Annotations (ZIP)")
            ann_file = gr.File(label="Annotations ZIP")
            export_ann_btn.click(fn=export_annotations, outputs=[ann_file])

            export_model_btn = gr.Button("📥 Export Trained Model")
            model_file = gr.File(label="Trained model (.pt)")
            export_model_btn.click(fn=export_model, outputs=[model_file])

    return demo


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        share=False,
        # Bind to all interfaces so the app is reachable from other machines
        # on the same network.  Change to "127.0.0.1" for local-only access.
        server_name="0.0.0.0",
        server_port=7860,
    )
