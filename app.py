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

import json
import logging
import re
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
# Lazy pipeline import
# ---------------------------------------------------------------------------

_pipeline: Optional[object] = None


def _get_pipeline():
    global _pipeline
    return _pipeline


def _set_pipeline(p):
    global _pipeline
    _pipeline = p


# ---------------------------------------------------------------------------
# Display-size constants – images are resized to fit within these dimensions
# so the <img> element has NO object-fit letterboxing.  Coordinates drawn
# on the display image are scaled back to original pixels via _session.scale.
# ---------------------------------------------------------------------------
DISPLAY_MAX_W = 960
DISPLAY_MAX_H = 640


def _render_image(image_path: str, annotations: List[Dict]) -> Optional["PIL.Image.Image"]:
    """Draw boxes on the image, then resize to display dimensions.

    Also updates ``_session.scale`` so drawn coordinates can be mapped
    back to origin pixel space.
    """
    try:
        from utils import draw_boxes
        from PIL import Image

        full = draw_boxes(image_path, annotations)
        orig_w, orig_h = full.size

        # Scale to fit display box while keeping aspect ratio
        ratio = min(DISPLAY_MAX_W / orig_w, DISPLAY_MAX_H / orig_h, 1.0)
        new_w = max(1, int(orig_w * ratio))
        new_h = max(1, int(orig_h * ratio))

        if ratio < 1.0:
            display = full.resize((new_w, new_h), Image.LANCZOS)
        else:
            display = full  # already small enough

        # Store scale so we can convert drawn coords → original coords
        _session.scale = orig_w / new_w  # same as orig_h / new_h
        return display
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
        self.annotations: Dict[str, List[Dict]] = {}
        self.pipeline_running: bool = False
        self.scale: float = 1.0  # display→original pixel multiplier

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
    """Markdown table of annotations."""
    if not anns:
        return "_No annotations for this image._"
    header = "| # | Class | Confidence | BBox (x1 y1 x2 y2) |\n|---|-------|-----------|---------------------|\n"
    rows = []
    for i, a in enumerate(anns):
        bbox = a.get("bbox", [])
        conf = a.get("confidence", "—")
        conf_str = f"{conf:.2f}" if isinstance(conf, float) else str(conf)
        bbox_str = ", ".join(f"{v:.0f}" for v in bbox) if bbox else "—"
        rows.append(f"| {i} | {a.get('class', '?')} | {conf_str} | {bbox_str} |")
    return header + "\n".join(rows)


def _annotation_choices(anns: List[Dict]) -> List[str]:
    """Choices for the CheckboxGroup selector."""
    choices = []
    for i, a in enumerate(anns):
        bbox = a.get("bbox", [])
        conf = a.get("confidence", "—")
        conf_str = f"{conf:.2f}" if isinstance(conf, float) else str(conf)
        bbox_str = ", ".join(f"{v:.0f}" for v in bbox) if bbox else "—"
        choices.append(f"#{i}: {a.get('class', '?')} ({conf_str}) [{bbox_str}]")
    return choices


def _parse_anns(annotation_json: str) -> List[Dict]:
    """Safely parse annotation JSON."""
    try:
        anns = json.loads(annotation_json)
        return anns if isinstance(anns, list) else []
    except (json.JSONDecodeError, TypeError):
        return _session.current_annotations()


def _save_to_pipeline(anns: List[Dict], img_path: str) -> None:
    """Persist annotations to the pipeline."""
    p = _get_pipeline()
    if p is None:
        return
    for ann in anns:
        cls = ann.get("class", "").strip()
        if cls:
            try:
                p.annotator.add_class(cls)
            except Exception:
                pass
    try:
        p.save_user_annotations(img_path, anns)
        p.mark_reviewed([img_path])
    except Exception as exc:
        logger.warning("Could not save annotations for %s: %s", img_path, exc)


def get_class_list() -> List[str]:
    p = _get_pipeline()
    if p is None:
        return []
    return p.annotator.classes


def review_load_current() -> Tuple:
    """Return (image, table, progress, json, classes, checkbox_update)."""
    img_path = _session.current_image()
    if img_path is None:
        return (
            None,
            "_No images to review._",
            "0 / 0",
            "[]",
            "none",
            gr.update(choices=[], value=[]),
        )
    anns = _session.current_annotations()
    rendered = _render_image(img_path, anns)
    table = _annotation_table(anns)
    progress = f"Image {_session.current_index + 1} / {len(_session.review_images)}"
    ann_json = json.dumps(anns, indent=2)
    classes = ", ".join(get_class_list()) or "none"
    choices = _annotation_choices(anns)
    return rendered, table, progress, ann_json, classes, gr.update(choices=choices, value=[])


def review_next() -> Tuple:
    if _session.current_index < len(_session.review_images) - 1:
        _session.current_index += 1
    return review_load_current()


def review_prev() -> Tuple:
    if _session.current_index > 0:
        _session.current_index -= 1
    return review_load_current()


def save_and_next(annotation_json: str) -> Tuple:
    """Save current annotations and advance."""
    img_path = _session.current_image()
    if img_path is None:
        return review_load_current()

    anns = _parse_anns(annotation_json)
    _session.set_current_annotations(anns)
    _save_to_pipeline(anns, img_path)
    return review_next()


def save_current(annotation_json: str) -> Tuple:
    """Save without advancing. Returns (status, image, table, json, checkbox_update)."""
    img_path = _session.current_image()
    if img_path is None:
        return (
            "No image loaded.",
            None,
            "_No images to review._",
            "[]",
            gr.update(choices=[], value=[]),
        )
    anns = _parse_anns(annotation_json)
    _session.set_current_annotations(anns)
    _save_to_pipeline(anns, img_path)

    rendered = _render_image(img_path, anns)
    choices = _annotation_choices(anns)
    return (
        f"✅ Saved {len(anns)} annotation(s).",
        rendered,
        _annotation_table(anns),
        json.dumps(anns, indent=2),
        gr.update(choices=choices, value=[]),
    )


def _edit_result(anns: List[Dict]) -> Tuple:
    """Common return after any edit operation: (image, table, json, checkbox_update)."""
    _session.set_current_annotations(anns)
    rendered = _render_image(_session.current_image(), anns)
    choices = _annotation_choices(anns)
    return (
        rendered,
        _annotation_table(anns),
        json.dumps(anns, indent=2),
        gr.update(choices=choices, value=[]),
    )


def delete_annotations(annotation_json: str, selected: List[str]) -> Tuple:
    """Remove all selected annotations (multi-select from CheckboxGroup)."""
    anns = _parse_anns(annotation_json)
    # Extract indices from strings like "#0: cat (0.85) [...]"
    indices = set()
    for s in (selected or []):
        m = re.match(r"#(\d+)", s)
        if m:
            indices.add(int(m.group(1)))
    # Remove in reverse order to keep indices valid
    for idx in sorted(indices, reverse=True):
        if 0 <= idx < len(anns):
            anns.pop(idx)
    return _edit_result(anns)


def add_annotation(annotation_json: str, new_class: str, coords_str: str) -> Tuple:
    """Add a bbox from drawn/typed coordinates.

    coords_str is "x1, y1, x2, y2" (comma-separated).
    """
    anns = _parse_anns(annotation_json)

    # Parse coordinates (these are in DISPLAY pixel space)
    parts = [p.strip() for p in coords_str.replace(";", ",").split(",") if p.strip()]
    if len(parts) < 4:
        return _edit_result(anns)
    try:
        x1, y1, x2, y2 = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
    except ValueError:
        return _edit_result(anns)

    # Scale display coords → original image coords
    s = _session.scale
    x1, y1, x2, y2 = x1 * s, y1 * s, x2 * s, y2 * s

    cls = new_class.strip() if new_class else "object"
    new_ann = {"bbox": [x1, y1, x2, y2], "class": cls, "confidence": 1.0}
    anns.append(new_ann)

    p = _get_pipeline()
    if p is not None and cls:
        try:
            p.annotator.add_class(cls)
        except Exception:
            pass

    return _edit_result(anns)


def change_classes(annotation_json: str, selected: List[str], new_class: str) -> Tuple:
    """Change the class for all selected annotations."""
    anns = _parse_anns(annotation_json)
    cls = new_class.strip() if new_class else ""
    if not cls:
        return _edit_result(anns)

    indices = set()
    for s in (selected or []):
        m = re.match(r"#(\d+)", s)
        if m:
            indices.add(int(m.group(1)))

    for idx in indices:
        if 0 <= idx < len(anns):
            anns[idx]["class"] = cls

    p = _get_pipeline()
    if p is not None and cls:
        try:
            p.annotator.add_class(cls)
        except Exception:
            pass

    return _edit_result(anns)


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
# JavaScript: canvas overlay for drawing bboxes on the image
# ===========================================================================

_BBOX_DRAW_JS = """
() => {
    // Version-gated: force re-init when code changes
    if (window._bboxVer === 3) return;
    window._bboxVer = 3;
    window._drawnCoords = '';

    // Remove any old overlay & listeners from previous version
    const oldOv = document.getElementById('bbox-overlay');
    if (oldOv) oldOv.remove();

    function setupOverlay() {
        const container = document.querySelector('#review-image-container');
        if (!container) { setTimeout(setupOverlay, 500); return; }

        const img = container.querySelector('img');
        if (!img || !img.complete || img.naturalWidth === 0) {
            setTimeout(setupOverlay, 300);
            return;
        }

        // Remove old overlay
        const old = document.getElementById('bbox-overlay');
        if (old) old.remove();

        // The Python side pre-resizes images so naturalWidth == displayed width.
        // We place the canvas exactly over the <img>, using getBoundingClientRect
        // for precise positioning.
        const overlay = document.createElement('canvas');
        overlay.id = 'bbox-overlay';
        overlay.style.pointerEvents = 'all';
        overlay.style.cursor = 'crosshair';
        overlay.style.zIndex = '100';
        overlay.style.position = 'absolute';

        const imgWrapper = img.parentElement;
        imgWrapper.style.position = 'relative';

        function positionCanvas() {
            const imgR = img.getBoundingClientRect();
            const wrapR = imgWrapper.getBoundingClientRect();
            overlay.style.left = (imgR.left - wrapR.left) + 'px';
            overlay.style.top  = (imgR.top  - wrapR.top)  + 'px';
            overlay.width  = Math.round(imgR.width);
            overlay.height = Math.round(imgR.height);
            overlay.style.width  = imgR.width  + 'px';
            overlay.style.height = imgR.height + 'px';
        }
        positionCanvas();
        imgWrapper.appendChild(overlay);

        const ctx = overlay.getContext('2d');
        let drawing = false, sx = 0, sy = 0;

        function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

        overlay.addEventListener('mousedown', (e) => {
            e.preventDefault();
            const r = overlay.getBoundingClientRect();
            sx = clamp(e.clientX - r.left, 0, overlay.width);
            sy = clamp(e.clientY - r.top,  0, overlay.height);
            drawing = true;
        });

        overlay.addEventListener('mousemove', (e) => {
            if (!drawing) return;
            const r = overlay.getBoundingClientRect();
            const cx = clamp(e.clientX - r.left, 0, overlay.width);
            const cy = clamp(e.clientY - r.top,  0, overlay.height);
            ctx.clearRect(0, 0, overlay.width, overlay.height);
            ctx.strokeStyle = '#00ff00';
            ctx.lineWidth = 2;
            ctx.setLineDash([6, 3]);
            ctx.strokeRect(sx, sy, cx - sx, cy - sy);
        });

        overlay.addEventListener('mouseup', (e) => {
            if (!drawing) return;
            drawing = false;
            const r = overlay.getBoundingClientRect();
            const ex = clamp(e.clientX - r.left, 0, overlay.width);
            const ey = clamp(e.clientY - r.top,  0, overlay.height);

            if (Math.abs(ex - sx) < 5 && Math.abs(ey - sy) < 5) {
                ctx.clearRect(0, 0, overlay.width, overlay.height);
                return;
            }

            // Since the image is pre-resized, naturalWidth == display width.
            // Simple scale: display-pixel coords in the resized image.
            const scX = img.naturalWidth  / overlay.width;
            const scY = img.naturalHeight / overlay.height;
            const x1 = Math.round(Math.min(sx, ex) * scX);
            const y1 = Math.round(Math.min(sy, ey) * scY);
            const x2 = Math.round(Math.max(sx, ex) * scX);
            const y2 = Math.round(Math.max(sy, ey) * scY);

            window._drawnCoords = x1 + ', ' + y1 + ', ' + x2 + ', ' + y2;

            // Persistent rectangle
            ctx.clearRect(0, 0, overlay.width, overlay.height);
            const dx = Math.min(sx, ex), dy = Math.min(sy, ey);
            const dw = Math.abs(ex - sx), dh = Math.abs(ey - sy);
            ctx.setLineDash([]);
            ctx.fillStyle = 'rgba(0, 255, 0, 0.12)';
            ctx.fillRect(dx, dy, dw, dh);
            ctx.strokeStyle = '#00ff00';
            ctx.lineWidth = 2;
            ctx.strokeRect(dx, dy, dw, dh);

            ctx.fillStyle = '#00ff00';
            ctx.font = 'bold 13px monospace';
            ctx.shadowColor = 'rgba(0,0,0,0.7)';
            ctx.shadowBlur = 3;
            const label = '[' + x1 + ', ' + y1 + ', ' + x2 + ', ' + y2 + ']';
            const ty = dy > 20 ? dy - 6 : dy + dh + 16;
            ctx.fillText(label, dx + 4, ty);
            ctx.shadowBlur = 0;

            // Update readonly coords textbox (best-effort)
            const coordsEl = document.querySelector('#coords-box textarea')
                          || document.querySelector('#coords-box input');
            if (coordsEl) {
                try {
                    const proto = coordsEl.tagName === 'TEXTAREA'
                        ? HTMLTextAreaElement : HTMLInputElement;
                    Object.getOwnPropertyDescriptor(proto.prototype, 'value')
                        .set.call(coordsEl, window._drawnCoords);
                    coordsEl.dispatchEvent(new Event('input', { bubbles: true }));
                } catch(err) {}
            }
        });

        new ResizeObserver(() => positionCanvas()).observe(img);
    }

    const observer = new MutationObserver(() => setTimeout(setupOverlay, 200));
    const target = document.querySelector('#review-image-container');
    if (target) {
        observer.observe(target, { childList: true, subtree: true, attributes: true });
    }
    setTimeout(setupOverlay, 500);
}
"""

# JS snippet for the Add button: injects window._drawnCoords into the
# coords input before the Python callback runs.
_ADD_BTN_JS = """
(ann_json, cls, coords) => {
    const drawn = window._drawnCoords;
    if (drawn && drawn.length > 0) {
        window._drawnCoords = '';  // consumed
        return [ann_json, cls, drawn];
    }
    return [ann_json, cls, coords];
}
"""


# ===========================================================================
# Build the Gradio UI
# ===========================================================================

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Auto-Annotation Pipeline") as demo:
        gr.Markdown(
            "# 🤖 Active Learning Auto-Annotation Pipeline\n\n"
            "Annotate object-detection datasets using pretrained YOLO + human-in-the-loop refinement."
        )

        # ---------------------------------------------------------------
        # Tab 1 – Setup
        # ---------------------------------------------------------------
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
                    minimum=0.05, maximum=0.95, step=0.05, value=0.25,
                )
            start_btn = gr.Button("🚀 Start Initial Detection", variant="primary")
            setup_log = gr.Textbox(label="Status", lines=6, interactive=False)
            start_btn.click(
                fn=setup_pipeline,
                inputs=[unlabeled_dir_input, model_choice, conf_slider],
                outputs=[setup_log],
            )

        # ---------------------------------------------------------------
        # Tab 2 – Review & Annotate
        # ---------------------------------------------------------------
        with gr.Tab("🖊️ Review & Annotate") as review_tab:
            gr.Markdown(
                "**🖱️ Draw a box** on the image by clicking & dragging. "
                "The rectangle stays visible and coordinates auto-fill. "
                "Type a class name and click **➕ Add**.\n\n"
                "**Select annotations** with the checkboxes below the table "
                "to **🗑 Delete** or **✏️ Change Class** in batch."
            )

            with gr.Row():
                progress_text = gr.Textbox(label="Progress", interactive=False, scale=1)
                class_list_display = gr.Textbox(label="Known classes", interactive=False, scale=3)

            # --- Image + Add controls + Table ---
            with gr.Row():
                with gr.Column(scale=3):
                    annotated_image = gr.Image(
                        label="Current image — click & drag to draw a box",
                        type="pil",
                        interactive=False,
                        elem_id="review-image-container",
                    )
                    # Add controls directly under the image
                    with gr.Row():
                        new_class = gr.Textbox(
                            label="Class name",
                            placeholder="e.g. cat",
                            scale=3,
                        )
                        coords_box = gr.Textbox(
                            label="Drawn coords",
                            placeholder="draw on image",
                            interactive=False,
                            elem_id="coords-box",
                            scale=2,
                        )
                        add_btn = gr.Button(
                            "➕ Add",
                            variant="primary",
                            scale=1,
                            min_width=80,
                        )
                with gr.Column(scale=2):
                    annotation_table_md = gr.Markdown("_No image loaded._")

            # --- Navigation ---
            with gr.Row():
                prev_btn = gr.Button("⬅ Previous")
                save_btn = gr.Button("💾 Save", variant="secondary")
                save_next_btn = gr.Button("💾 Save & Next", variant="primary")
                next_btn = gr.Button("Next ➡")

            save_status = gr.Textbox(label="Status", interactive=False, lines=1)

            # --- Selection-based edit / delete ---
            gr.Markdown("---")
            gr.Markdown(
                "#### ✏️ Edit / 🗑 Delete\n"
                "*Select one or more annotations below, then click Delete or Change Class.*"
            )
            ann_checkbox = gr.CheckboxGroup(
                label="Select annotations",
                choices=[],
                value=[],
                elem_id="ann-selector",
            )
            with gr.Row():
                chg_class = gr.Textbox(label="New class name", placeholder="dog", scale=2)
                chg_btn = gr.Button("✏️ Change Class of Selected", scale=1)
                del_btn = gr.Button("🗑 Delete Selected", variant="stop", scale=1)

            # --- Raw JSON (advanced) ---
            annotation_json = gr.Textbox(
                label="Annotations JSON (advanced — editable)",
                lines=6,
                visible=True,
            )

            # --- Wire events ---
            _review_outputs = [
                annotated_image,
                annotation_table_md,
                progress_text,
                annotation_json,
                class_list_display,
                ann_checkbox,
            ]

            review_tab.select(fn=review_load_current, outputs=_review_outputs)

            load_btn = gr.Button("🔄 Reload current image")
            load_btn.click(fn=review_load_current, outputs=_review_outputs)
            prev_btn.click(fn=review_prev, outputs=_review_outputs)
            next_btn.click(fn=review_next, outputs=_review_outputs)
            save_next_btn.click(fn=save_and_next, inputs=[annotation_json], outputs=_review_outputs)

            _save_outputs = [save_status, annotated_image, annotation_table_md, annotation_json, ann_checkbox]
            save_btn.click(fn=save_current, inputs=[annotation_json], outputs=_save_outputs)

            _edit_outputs = [annotated_image, annotation_table_md, annotation_json, ann_checkbox]
            del_btn.click(
                fn=delete_annotations,
                inputs=[annotation_json, ann_checkbox],
                outputs=_edit_outputs,
            )
            chg_btn.click(
                fn=change_classes,
                inputs=[annotation_json, ann_checkbox, chg_class],
                outputs=_edit_outputs,
            )
            add_btn.click(
                fn=add_annotation,
                inputs=[annotation_json, new_class, coords_box],
                outputs=_edit_outputs,
                js=_ADD_BTN_JS,
            )

            refresh_classes_btn = gr.Button("🔄 Refresh class list")
            refresh_classes_btn.click(
                fn=lambda: ", ".join(get_class_list()) or "none",
                outputs=[class_list_display],
            )

            # Inject the canvas-overlay drawing JS
            demo.load(fn=None, js=_BBOX_DRAW_JS)

        # ---------------------------------------------------------------
        # Tab 3 – Training & Auto-Annotation
        # ---------------------------------------------------------------
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

            train_btn.click(fn=run_training, inputs=[epochs_input, device_input], outputs=[train_output])
            auto_btn.click(fn=run_auto_annotation, outputs=[auto_output])
            summary_btn.click(fn=pipeline_summary, outputs=[summary_output])

        # ---------------------------------------------------------------
        # Tab 4 – Export
        # ---------------------------------------------------------------
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
        server_name="127.0.0.1",
        server_port=7860,
    )
