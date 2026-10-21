"""Loss and COCO-style mAP (torchmetrics) on any split."""
from __future__ import annotations

from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm.auto import tqdm

from .utils import move_labels_to_device, post_process, scale_normalized_boxes


def autocast_context(device: torch.device, fp16: bool):
    if fp16 and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    image_processor,
    loader: DataLoader,
    device: torch.device,
    id2label: dict[int, str],
    threshold: float = 0.0,
    use_nms: bool = False,
    nms_iou: float = 0.5,
    fp16: bool = False,
    desc: str = "eval",
) -> dict[str, float]:
    """Average loss + mAP/mAR (overall and per class)
    """
    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
    loss_sum, n_images = 0.0, 0
    loss_parts: dict[str, float] = {}

    for batch in tqdm(loader, desc=desc, leave=False):
        pixel_values = batch["pixel_values"].to(device)
        labels = move_labels_to_device(batch["labels"], device)

        with autocast_context(device, fp16):
            outputs = model(pixel_values=pixel_values, labels=labels)

        bs = pixel_values.shape[0]
        loss_sum += float(outputs.loss) * bs
        n_images += bs
        for k, v in (outputs.loss_dict or {}).items():
            loss_parts[k] = loss_parts.get(k, 0.0) + float(v) * bs

        target_sizes = torch.stack([t["orig_size"] for t in batch["labels"]])

        preds = post_process(image_processor, outputs.logits.cpu(), outputs.pred_boxes.cpu(),
                             target_sizes, threshold=threshold, use_nms=use_nms, nms_iou=nms_iou)
        targets = [
            {"boxes": scale_normalized_boxes(t["boxes"], size), "labels": t["class_labels"]}
            for t, size in zip(batch["labels"], target_sizes)
        ]
        metric.update(
            [{"boxes": p["boxes"].float(), "scores": p["scores"].float(), "labels": p["labels"]} for p in preds],
            targets,
        )

    results = metric.compute()
    classes = results.pop("classes")
    map_per_class = results.pop("map_per_class")
    mar_per_class = results.pop("mar_100_per_class")
    metrics = {k: round(float(v), 4) for k, v in results.items()}

    if classes.ndim == 0:  # a single class present
        classes, map_per_class, mar_per_class = classes.unsqueeze(0), map_per_class.unsqueeze(0), mar_per_class.unsqueeze(0)
    for cid, m, r in zip(classes.tolist(), map_per_class.tolist(), mar_per_class.tolist()):
        name = id2label[int(cid)]
        metrics[f"map_{name}"] = round(float(m), 4)
        metrics[f"mar_100_{name}"] = round(float(r), 4)

    metrics["loss"] = round(loss_sum / max(n_images, 1), 4)
    for k, v in loss_parts.items():
        metrics[k] = round(v / max(n_images, 1), 4)
    return metrics


def format_metrics(metrics: dict[str, float], prefix: str = "") -> str:
    main = ["loss", "map", "map_50", "map_75", "map_small", "map_medium", "map_large", "mar_100"]
    lines = [f"{prefix}{k:<12}: {metrics[k]:.4f}" for k in main if k in metrics]
    per_class = [k for k in metrics if k.startswith("map_") and k not in main]
    if per_class:
        lines.append(f"{prefix}per-class mAP: " + ", ".join(f"{k[4:]}={metrics[k]:.4f}" for k in per_class))
    return "\n".join(lines)
