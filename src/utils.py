"""Shared helpers: seeding, devices, image loading, augmentation, collation, box maths and plotting."""
from __future__ import annotations

import io
import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import albumentations as A
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import requests
import torch
from PIL import Image, ImageDraw
from torchvision.ops import batched_nms

LOGGER_NAME = "detr_cppe5"


# ════════════════════════════════════════════════════════════════════════
#  General
# ════════════════════════════════════════════════════════════════════════
def get_logger() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        return f"CUDA: {torch.cuda.get_device_name(0)} ({props.total_memory / 1e9:.1f} GB VRAM)"
    if device.type == "mps":
        return "Apple MPS"
    return "CPU (no GPU detected — training will be slow)"


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(o: Any) -> Any:
    if isinstance(o, torch.Tensor):
        return o.tolist()
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not JSON serialisable: {type(o)}")


def load_image(source: str | Path) -> Image.Image:
    """Load an RGB image from a local path or an http(s) URL."""
    source = str(source)
    if source.startswith(("http://", "https://")):
        resp = requests.get(source, timeout=30, headers={"User-Agent": "Mozilla/5.0 (detr-cppe5)"})
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    return Image.open(source).convert("RGB")


# ════════════════════════════════════════════════════════════════════════
#  Annotations & augmentation
# ════════════════════════════════════════════════════════════════════════
def sanitize_coco_boxes(
    bboxes: Sequence[Sequence[float]], categories: Sequence[int], width: int, height: int, min_size: float = 1.0
) -> tuple[list[list[float]], list[int]]:
    """Clip COCO [x, y, w, h] boxes to the image and drop degenerate ones (albumentations rejects them)."""
    clean_boxes, clean_cats = [], []
    for (x, y, w, h), c in zip(bboxes, categories):
        x0, y0 = max(0.0, float(x)), max(0.0, float(y))
        x1, y1 = min(float(width), float(x) + float(w)), min(float(height), float(y) + float(h))
        if x1 - x0 >= min_size and y1 - y0 >= min_size:
            clean_boxes.append([x0, y0, x1 - x0, y1 - y0])
            clean_cats.append(int(c))
    return clean_boxes, clean_cats


def build_transforms(train: bool) -> A.Compose:
    """Albumentations pipelines (boxes in COCO format, labels in the 'category' field)."""
    if train:
        return A.Compose(
            [
                A.Perspective(p=0.1),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(p=0.1),
            ],
            bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True, min_area=25),
        )
    return A.Compose(
        [A.NoOp()],
        bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True),
    )


def format_image_annotations_as_coco(
    image_id: int, categories: Sequence[int], bboxes: Sequence[Sequence[float]]
) -> dict:
    """Build the COCO-style annotation dict the DETR image processors expect."""
    annotations = [
        {
            "image_id": image_id,
            "category_id": int(c),
            "iscrowd": 0,
            "area": float(b[2]) * float(b[3]),
            "bbox": [float(v) for v in b],
        }
        for c, b in zip(categories, bboxes)
    ]
    return {"image_id": image_id, "annotations": annotations}


# ════════════════════════════════════════════════════════════════════════
#  Batching
# ════════════════════════════════════════════════════════════════════════
def collate_fn(batch: list[dict]) -> dict:
    """Pad images to the largest one in the batch and keep boxes consistent with the padded canvas.

    The image processors normalise boxes (cx, cy, w, h) relative to the tensor they return. When images
    of different sizes are padded together, the boxes are re-normalised to the new canvas so targets and
    predictions always refer to the same frame.
    """
    heights = [item["pixel_values"].shape[-2] for item in batch]
    widths = [item["pixel_values"].shape[-1] for item in batch]
    H, W = max(heights), max(widths)

    pixel_values = torch.zeros(len(batch), 3, H, W, dtype=batch[0]["pixel_values"].dtype)
    pixel_mask = torch.zeros(len(batch), H, W, dtype=torch.long)
    labels = []
    for i, item in enumerate(batch):
        h, w = item["pixel_values"].shape[-2:]
        pixel_values[i, :, :h, :w] = item["pixel_values"]
        pixel_mask[i, :h, :w] = item["pixel_mask"]
        target = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in item["labels"].items()}
        if (h, w) != (H, W) and target["boxes"].numel():
            scale = torch.tensor([w / W, h / H, w / W, h / H], dtype=target["boxes"].dtype)
            target["boxes"] = target["boxes"] * scale
        target["size"] = torch.tensor([H, W])
        labels.append(target)
    return {"pixel_values": pixel_values, "pixel_mask": pixel_mask, "labels": labels}


def move_labels_to_device(labels: list[dict], device: torch.device) -> list[dict]:
    return [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in labels]


# ════════════════════════════════════════════════════════════════════════
#  Boxes
# ════════════════════════════════════════════════════════════════════════
def center_to_corners(boxes: torch.Tensor) -> torch.Tensor:
    """(cx, cy, w, h) -> (x0, y0, x1, y1)."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def scale_normalized_boxes(boxes_cxcywh: torch.Tensor, size_hw: torch.Tensor) -> torch.Tensor:
    """Normalised (cx, cy, w, h) -> absolute (x0, y0, x1, y1) for an image of size (h, w)."""
    h, w = size_hw.float().unbind(-1)
    return center_to_corners(boxes_cxcywh.float()) * torch.stack([w, h, w, h])


def apply_nms(result: dict, iou_threshold: float) -> dict:
    """Class-wise NMS on one post-processed result dict (scores / labels / boxes)."""
    if result["boxes"].numel() == 0:
        return result
    keep = batched_nms(result["boxes"].float(), result["scores"].float(), result["labels"], iou_threshold)
    return {k: v[keep] for k, v in result.items()}


def post_process(
    image_processor,
    logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    target_sizes: torch.Tensor,
    threshold: float,
    use_nms: bool = False,
    nms_iou: float = 0.5,
) -> list[dict]:
    """Run the model's own post-processing (+ optional NMS) on raw outputs from PyTorch or ONNX."""

    class _Outputs:  # the processors only read .logits and .pred_boxes
        def __init__(self, lg: torch.Tensor, pb: torch.Tensor) -> None:
            self.logits, self.pred_boxes = lg, pb

    results = image_processor.post_process_object_detection(
        _Outputs(logits.float(), pred_boxes.float()), threshold=threshold, target_sizes=target_sizes
    )
    if use_nms:
        results = [apply_nms(r, nms_iou) for r in results]
    return results


# ════════════════════════════════════════════════════════════════════════
#  Plotting
# ════════════════════════════════════════════════════════════════════════
def _finish_plot(fig, save_path: str | Path | None, show: bool) -> None:
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def count_labels(object_lists: Iterable[dict], id2label: dict[int, str]) -> dict[str, int]:
    counts = Counter(c for objs in object_lists for c in objs["category"])
    return {id2label[k]: counts.get(k, 0) for k in sorted(id2label)}


def plot_class_distribution(
    counts: dict[str, int], title: str = "Number of Occurrences by Category",
    save_path: str | Path | None = None, show: bool = False,
) -> None:
    """Bar chart of box counts per category, sorted from most to least frequent."""
    data = dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))
    names, values = list(data), list(data.values())
    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
    bars = ax.bar(names, values, color="skyblue", edgecolor="none")
    top = max(values) if values else 1
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height + top * 0.01, f"{int(height)}",
                ha="center", va="bottom", fontsize=9)
    ax.set_title(title, fontsize=16, pad=15)
    ax.set_xlabel("Categories", fontsize=14, labelpad=10)
    ax.set_ylabel("Number of Occurrences", fontsize=14, labelpad=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=10)
    _finish_plot(fig, save_path, show)


def draw_detections(
    image: Image.Image,
    boxes: Sequence[Sequence[float]],
    labels: Sequence[str],
    scores: Sequence[float] | None = None,
    color: str = "red",
) -> Image.Image:
    """Draw (x0, y0, x1, y1) boxes with their label (and score) on a copy of the image."""
    image = image.convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    for i, (box, label) in enumerate(zip(boxes, labels)):
        x0, y0, x1, y1 = (float(v) for v in box)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        text = label if scores is None else f"{label}: {float(scores[i]):.2f}"
        tw = draw.textlength(text)
        draw.rectangle([x0, y0, x0 + tw + 6, y0 + 12], fill=color)
        draw.text((x0 + 3, y0), text, fill="white")
    return image


def draw_sample(dataset_split, idx: int, id2label: dict[int, str]) -> tuple[Image.Image, int, list[int]]:
    """Ground-truth boxes of one dataset sample (COCO boxes)."""
    sample = dataset_split[idx]
    annotations = sample["objects"]
    boxes = [[x, y, x + w, y + h] for x, y, w, h in annotations["bbox"]]
    image = draw_detections(sample["image"], boxes, [id2label[c] for c in annotations["category"]])
    return image, len(boxes), list(annotations["category"])


def plot_training_curves(
    history: list[dict], metric: str, best_epoch: int | None,
    save_path: str | Path | None = None, show: bool = False,
) -> None:
    """Train/validation loss and the validation metric used to select the best model."""
    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5), dpi=100)
    ax1.plot(epochs, [h["train_loss"] for h in history], marker="o", label="train loss")
    ax1.plot(epochs, [h["val_loss"] for h in history], marker="o", label="validation loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.set_title("Loss"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(epochs, [h[f"val_{metric}"] for h in history], marker="o", color="tab:green", label=f"val {metric}")
    if best_epoch is not None:
        best = next(h for h in history if h["epoch"] == best_epoch)
        ax2.scatter([best_epoch], [best[f"val_{metric}"]], s=150, facecolors="none", edgecolors="red",
                    linewidths=2, label=f"best (epoch {best_epoch})", zorder=3)
    ax2.set_xlabel("epoch"); ax2.set_ylabel(metric); ax2.set_title(f"Validation {metric}")
    ax2.legend(); ax2.grid(alpha=0.3)
    for ax in (ax1, ax2):
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    _finish_plot(fig, save_path, show)
