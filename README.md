# Auto-Annotation — Active Learning Pipeline for Object Detection

An end-to-end **active learning** system that uses a pretrained YOLOv8 model to
bootstrap bounding-box annotations, lets you correct them through an interactive
Gradio web UI, fine-tunes the model on your corrections, and iterates until you
are satisfied with the annotation quality.

---

## Architecture

```
Unlabeled Images ──► YOLO Detect (pretrained) ──► User Reviews & Corrects (Gradio UI)
      ▲                                                         │
      │                                                  Fine-tuned YOLO
      │                                                         │
      └──── Pick 100 least confident ◄──── Auto-annotate 1 000 images
                 (user corrects, retrain, repeat)
```

---

## Features

| Feature | Details |
|---|---|
| **Pretrained bootstrap** | Uses YOLOv8n/s/m/l/x (COCO) for zero-shot detection |
| **Interactive annotation UI** | Accept, change class, delete, or add bounding boxes |
| **Uncertainty sampling** | Selects the *least confident* images for human review |
| **Active learning loop** | Auto-annotate → select uncertain → correct → retrain |
| **YOLO format output** | `.txt` label files compatible with Ultralytics training |
| **Resumable** | Pipeline state is persisted to disk after every save |
| **Configurable** | All parameters have sensible defaults, all are overridable |

---

## Project Structure

```
auto_annotation/
├── app.py               # Main Gradio application (single entry point)
├── pipeline.py          # Active learning loop orchestration
├── detector.py          # YOLO detection wrapper
├── annotator.py         # YOLO label file I/O + class management
├── active_learning.py   # Uncertainty sampling & auto-annotation helpers
├── trainer.py           # Fine-tuning wrapper around ultralytics
├── utils.py             # Image I/O, drawing, directory helpers
├── requirements.txt     # Python dependencies
├── dataset.yaml         # Template dataset config (auto-updated at runtime)
└── data/
    ├── images/
    │   ├── train/       # Images copied here before training
    │   └── val/
    ├── labels/
    │   ├── train/       # YOLO .txt label files
    │   └── val/
    └── unlabeled/       # ← DROP YOUR IMAGES HERE
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/othmaneirl/auto_annotation.git
cd auto_annotation

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **GPU support** — install PyTorch with CUDA before running pip install, or
> pass `device="0"` in the Training tab.

---

## Quick Start

```bash
# 1. Put your unlabeled images in data/unlabeled/
cp /path/to/my/images/*.jpg data/unlabeled/

# 2. Launch the app
python app.py
# → Opens at http://localhost:7860
```

### Workflow

1. **Setup & Import tab**
   - Enter the path to your unlabeled images directory.
   - Choose a base YOLO model (`yolov8n.pt` is fast, `yolov8x.pt` is most accurate).
   - Set the confidence threshold.
   - Click **Start Initial Detection** — the pretrained model runs on all images.

2. **Review & Annotate tab**
   - Click **Load current image** to display the first image with detected boxes.
   - Use the JSON editor to accept, change, or delete boxes.
   - Click **Add Annotation** to draw in missing boxes.
   - Click **Save & Next** to persist changes and move to the next image.

3. **Training & Auto-Annotation tab**
   - Click **Fine-tune Model** to train on your corrected annotations.
   - Click **Run Auto-Annotation** to let the freshly trained model annotate the
     remaining images and select the most uncertain ones for your next review batch.
   - Repeat steps 2–3 until quality is satisfactory.

4. **Export tab**
   - Download all annotations as a ZIP archive.
   - Download the latest trained model (`.pt`).

---

## How the Active Learning Loop Works

```
Round 0:
    pretrained model → detect all unlabeled images
    select 100 least-confident → user corrects → save labels
    fine-tune on corrected data → save model v1

Round 1:
    model v1 → auto-annotate up to 1 000 images
    select 100 least-confident → user corrects → save labels
    fine-tune on all corrected data → save model v2

Round 2: ... (repeat until satisfied)
```

**Uncertainty score** for an image = average detection confidence of all boxes.  
Images with *no* detections score 0.0 (maximum uncertainty).

---

## Configuration

All parameters are set via `PipelineConfig` in `pipeline.py` (or the UI):

| Parameter | Default | Description |
|---|---|---|
| `unlabeled_dir` | `data/unlabeled` | Directory of raw images |
| `base_model` | `yolov8n.pt` | Starting YOLO weights |
| `conf_threshold` | `0.25` | Detection confidence cutoff |
| `batch_size_auto_annotate` | `1000` | Images auto-annotated per round |
| `batch_size_review` | `100` | Uncertain images shown to the user |
| `training_epochs_first` | `50` | Epochs for round 0 |
| `training_epochs_subsequent` | `30` | Epochs for later rounds |
| `image_size` | `640` | Training image resolution |
| `val_ratio` | `0.1` | Fraction of data used for validation |

---

## Example Workflow

```
# Start with 500 unlabeled images of cats and dogs
cp ~/datasets/pets/*.jpg data/unlabeled/
python app.py

# In the UI:
# 1. Setup tab → model=yolov8n.pt, threshold=0.25 → Start Detection
# 2. Annotate tab → correct 100 images (delete wrong boxes, add missing ones)
# 3. Training tab → Fine-tune (50 epochs) → Run Auto-Annotation
# 4. Annotate tab → correct the next 100 uncertain images
# 5. Training tab → Fine-tune (30 epochs) → Run Auto-Annotation
# 6. Export tab → download annotations.zip + best.pt
```

---

## Data Directory

The `data/` directory is **not committed** (see `.gitignore`). Its structure:

```
data/
├── classes.txt          # One class name per line (auto-managed)
├── dataset.yaml         # Auto-generated Ultralytics dataset config
├── pipeline_state.json  # Resumability checkpoint
├── images/train/        # Symlinks / copies of reviewed training images
├── images/val/
├── labels/train/        # YOLO .txt label files
├── labels/val/
└── unlabeled/           # Raw images (untouched)
```

---

## Tech Stack

- **[ultralytics](https://github.com/ultralytics/ultralytics)** — YOLOv8 detection & training
- **[Gradio](https://gradio.app)** — web UI
- **[Pillow](https://pillow.readthedocs.io)** — image I/O & rendering
- **[OpenCV](https://opencv.org)** — optional image processing
- **[PyYAML](https://pyyaml.org)** — dataset config generation
- **[NumPy](https://numpy.org)** — numerical operations
