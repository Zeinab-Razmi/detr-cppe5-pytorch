"""Predictions with the best saved weights: on an image (path or URL) or on test-split images."""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from .config import Config
from .evaluate import autocast_context
from .model import load_trained
from .utils import draw_detections, get_device, get_logger, load_image, post_process, save_json

log = get_logger()


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    image_processor,
    image: Image.Image,
    device: torch.device,
    threshold: float,
    use_nms: bool = False,
    nms_iou: float = 0.5,
    fp16: bool = False,
) -> list[dict]:
    """Detections on one PIL image as [{'label', 'label_id', 'score', 'box': [x0, y0, x1, y1]}] (original pixels).
    """
    model.eval()
    inputs = image_processor(images=[image], return_tensors="pt").to(device)
    with autocast_context(device, fp16):
        outputs = model(**inputs)

    target_sizes = torch.tensor([[image.height, image.width]])
    result = post_process(image_processor, outputs.logits.cpu(), outputs.pred_boxes.cpu(), target_sizes,
                          threshold=threshold, use_nms=use_nms, nms_iou=nms_iou)[0]
    return detections_to_list(result, model.config.id2label, image.size)


def detections_to_list(result: dict, id2label: dict, image_size: tuple[int, int]) -> list[dict]:
    """Post-processed tensors -> sorted list of detections with boxes clipped to the image."""
    w, h = image_size
    dets = []
    for score, label, box in zip(result["scores"].tolist(), result["labels"].tolist(), result["boxes"].tolist()):
        x0, y0, x1, y1 = box
        box = [round(max(0.0, min(w, x0)), 2), round(max(0.0, min(h, y0)), 2),
               round(max(0.0, min(w, x1)), 2), round(max(0.0, min(h, y1)), 2)]
        dets.append({"label": id2label[int(label)],
                     "label_id": int(label), "score": round(float(score), 4), "box": box})
    return sorted(dets, key=lambda d: d["score"], reverse=True)


def print_detections(dets: list[dict], source: str = "", max_lines: int = 20) -> None:
    log.info(f"{len(dets)} detections{f' in {source}' if source else ''}")
    for d in dets[:max_lines]:
        log.info(f"  Detected {d['label']} with confidence {d['score']:.3f} at location {d['box']}")
    if len(dets) > max_lines:
        log.info(f"  ... {len(dets) - max_lines} more (all saved to the JSON file)")


def predict_image(cfg: Config, source: str | None = None, threshold: float | None = None) -> list[dict]:
    """Run the best model on an image path/URL (default: inference.image), save the drawing + JSON."""
    source = source or cfg.inference["image"]
    threshold = cfg.model.inference_threshold if threshold is None else threshold
    device = get_device()
    model, processor = load_trained(cfg, device)
    image = load_image(source)

    dets = predict(model, processor, image, device, threshold,
                   use_nms=cfg.model.use_nms, nms_iou=cfg.model.nms_iou, fp16=cfg.model.fp16 and device.type == "cuda")
    print_detections(dets, str(source))

    out_dir = cfg.output_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(str(source).split("?")[0]).stem or "image"
    drawn = draw_detections(image, [d["box"] for d in dets], [d["label"] for d in dets], [d["score"] for d in dets])
    drawn.save(out_dir / f"{stem}_pred.jpg")
    save_json({"source": str(source), "threshold": threshold, "nms": cfg.model.use_nms, "detections": dets},
              out_dir / f"{stem}_pred.json")
    log.info(f"Saved {out_dir / f'{stem}_pred.jpg'}")
    if cfg.output.get("show_plots", False):
        _show(drawn)
    return dets


def select_indices(cfg: Config, split_size: int, indices: list[int] | None = None,
                   num_images: int | None = None) -> list[int]:
    """Which split images to run on: explicit indices (CLI, then config), else the first N images."""
    chosen = indices if indices else cfg.inference.get("test_indices") or []
    if chosen:
        bad = [i for i in chosen if not 0 <= int(i) < split_size]
        if bad:
            raise IndexError(f"Image indices {bad} out of range: the split has {split_size} images (0-{split_size - 1})")
        return [int(i) for i in chosen]
    n = int(num_images if num_images is not None else cfg.inference["num_test_images"])
    return list(range(min(n, split_size)))


def predict_split(cfg: Config, hf_split, num_images: int | None = None, split_name: str = "test",
                  threshold: float | None = None, indices: list[int] | None = None) -> list[dict]:
    """Predictions (red) next to ground truth (green) for chosen images of a split."""
    chosen = select_indices(cfg, len(hf_split), indices, num_images)
    threshold = cfg.model.inference_threshold if threshold is None else threshold
    device = get_device()
    model, processor = load_trained(cfg, device)
    id2label = model.config.id2label
    out_dir = cfg.output_dir / "predictions" / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for idx in chosen:
        example = hf_split[idx]
        image = example["image"].convert("RGB")
        dets = predict(model, processor, image, device, threshold,
                       use_nms=cfg.model.use_nms, nms_iou=cfg.model.nms_iou,
                       fp16=cfg.model.fp16 and device.type == "cuda")
        gt_boxes = [[x, y, x + w, y + h] for x, y, w, h in example["objects"]["bbox"]]
        gt_labels = [id2label[int(c)] for c in example["objects"]["category"]]
        gt = draw_detections(image, gt_boxes, gt_labels, color="green")
        pred = draw_detections(image, [d["box"] for d in dets], [d["label"] for d in dets], [d["score"] for d in dets])
        canvas = Image.new("RGB", (image.width * 2 + 10, image.height), "white")
        canvas.paste(gt, (0, 0))
        canvas.paste(pred, (image.width + 10, 0))
        canvas.save(out_dir / f"{split_name}_{idx:03d}_gt_vs_pred.jpg")
        all_results.append({"index": idx, "image_id": int(example["image_id"]), "detections": dets})
        log.info(f"[{split_name} #{idx}] ground truth {len(gt_boxes)} boxes | predicted {len(dets)}")
    save_json(all_results, out_dir / "predictions.json")
    log.info(f"Saved {len(all_results)} images (left: ground truth, right: prediction) to {out_dir}")
    return all_results


def _show(image: Image.Image) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 7))
    plt.imshow(image)
    plt.axis("off")
    plt.show()
