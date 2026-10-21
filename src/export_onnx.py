"""Export the best weights to ONNX and check ONNX Runtime against PyTorch on the same inputs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image

from .config import Config
from .inference import detections_to_list, print_detections
from .model import load_trained
from .utils import draw_detections, get_logger, load_image, post_process, save_json

log = get_logger()


class OnnxDetector(torch.nn.Module):
    """Plain-tensor wrapper: pixel_values -> (logits, pred_boxes), used when the model was trained without a mask."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor):
        out = self.model(pixel_values=pixel_values, return_dict=True)
        return out.logits, out.pred_boxes


class OnnxDetectorWithMask(OnnxDetector):
    """Plain-tensor wrapper: (pixel_values, pixel_mask) -> (logits, pred_boxes)."""

    def forward(self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor):
        out = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, return_dict=True)
        return out.logits, out.pred_boxes


def export_onnx(cfg: Config) -> Path:
    """Write cfg.onnx_path (dynamic batch / height / width) and validate it with onnx.checker."""
    model, processor = load_trained(cfg, torch.device("cpu"))
    model = model.float().eval()
    size = int(cfg.data["image_size"])
    # Non-square dummy: the processor pads it, so the traced graph includes the padding-mask path.
    dummy = processor(images=[Image.new("RGB", (size, int(size * 0.6)))], return_tensors="pt")
    pixel_values = dummy["pixel_values"]
    pixel_mask = dummy.get("pixel_mask", torch.ones(1, *pixel_values.shape[-2:], dtype=torch.long))

    path = cfg.onnx_path
    path.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Exporting {cfg.model.name} (fine-tuned) to ONNX, opset {cfg.export['opset']} ...")
    wrapper, args, input_names = OnnxDetectorWithMask(model), (pixel_values, pixel_mask), ["pixel_values", "pixel_mask"]
    dynamic_axes = {
        "pixel_values": {0: "batch", 2: "height", 3: "width"},
        "pixel_mask": {0: "batch", 1: "height", 2: "width"},
        "logits": {0: "batch"},
        "pred_boxes": {0: "batch"},
    }
    torch.onnx.export(
        wrapper,
        args,
        str(path),
        input_names=input_names,
        output_names=["logits", "pred_boxes"],
        dynamic_axes={k: v for k, v in dynamic_axes.items() if k in input_names + ["logits", "pred_boxes"]},
        opset_version=int(cfg.export["opset"]),
        do_constant_folding=True,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(str(path)))
    processor.save_pretrained(path.parent)  # pre/post-processing settings travel with the .onnx file
    log.info(f"Saved {path} ({path.stat().st_size / 1e6:.1f} MB) — onnx.checker passed")
    return path


def onnx_session(path: str | Path) -> ort.InferenceSession:
    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
    return ort.InferenceSession(str(path), providers=providers)


def onnx_predict_raw(session: ort.InferenceSession, inputs: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Raw ONNX outputs for processor outputs (tensors) -> (logits, pred_boxes) as torch tensors."""
    pixel_values = inputs["pixel_values"]
    feed = {"pixel_values": pixel_values.numpy().astype(np.float32)}
    if "pixel_mask" in {i.name for i in session.get_inputs()}:
        pixel_mask = inputs.get("pixel_mask", torch.ones(pixel_values.shape[0], *pixel_values.shape[-2:], dtype=torch.long))
        feed["pixel_mask"] = pixel_mask.numpy().astype(np.int64)
    logits, boxes = session.run(["logits", "pred_boxes"], feed)
    return torch.from_numpy(logits), torch.from_numpy(boxes)


def _match_detections(a: list[dict], b: list[dict]) -> dict:
    """Pair every PyTorch detection with the closest unused ONNX detection of the same class.
    Matching (instead of comparing by rank) keeps the check stable when two detections have
    near-identical scores and swap order because of float rounding.
    """
    unused = list(range(len(b)))
    box_diff, score_diff, unmatched = 0.0, 0.0, 0
    for d in a:
        candidates = [j for j in unused if b[j]["label_id"] == d["label_id"]]
        if not candidates:
            unmatched += 1
            continue
        j = min(candidates, key=lambda k: sum(abs(x - y) for x, y in zip(d["box"], b[k]["box"])))
        unused.remove(j)
        box_diff = max(box_diff, max(abs(x - y) for x, y in zip(d["box"], b[j]["box"])))
        score_diff = max(score_diff, abs(d["score"] - b[j]["score"]))
    return {
        "count_pytorch": len(a),
        "count_onnx": len(b),
        "same_count": len(a) == len(b),
        "all_matched": unmatched == 0 and not unused,
        "max_box_diff_px": round(float(box_diff), 4),
        "max_score_diff": round(float(score_diff), 6),
    }


@torch.no_grad()
def compare_with_pytorch(cfg: Config, onnx_path: str | Path | None = None, test_split=None, num_test_images: int = 4) -> dict:
    """Run PyTorch and ONNX Runtime on the inference image (+ some test images) and compare outputs."""
    onnx_path = Path(onnx_path or cfg.onnx_path)
    model, processor = load_trained(cfg, torch.device("cpu"))
    model = model.float().eval()
    session = onnx_session(onnx_path)
    atol = float(cfg.export["atol"])
    threshold = cfg.model.inference_threshold

    images: list[tuple[str, Image.Image]] = []
    try:
        images.append(("inference_image", load_image(cfg.inference["image"])))
    except Exception as exc:  # no internet, broken URL ...
        log.warning(f"Could not load inference.image ({exc}); comparing on test images only.")
    if test_split is not None:
        for i in range(min(num_test_images, len(test_split))):
            images.append((f"test_{i}", test_split[i]["image"].convert("RGB")))
    if not images:
        raise RuntimeError("No images available to compare ONNX and PyTorch outputs.")

    out_dir = onnx_path.parent
    report = {"onnx_path": str(onnx_path), "atol": atol, "threshold": threshold, "nms": cfg.model.use_nms, "images": []}
    for name, image in images:
        inputs = processor(images=[image], return_tensors="pt")
        mask = inputs.get("pixel_mask", torch.ones(1, *inputs["pixel_values"].shape[-2:], dtype=torch.long))
        pt_out = model(pixel_values=inputs["pixel_values"], pixel_mask=mask)  # = predict(): model(**inputs)
        ox_logits, ox_boxes = onnx_predict_raw(session, inputs)

        logits_diff = float((pt_out.logits - ox_logits).abs().max())
        boxes_diff = float((pt_out.pred_boxes - ox_boxes).abs().max())

        target_sizes = torch.tensor([[image.height, image.width]])  # as in the notebook
        kw = dict(threshold=threshold, use_nms=cfg.model.use_nms, nms_iou=cfg.model.nms_iou)
        pt_dets = detections_to_list(post_process(processor, pt_out.logits, pt_out.pred_boxes, target_sizes, **kw)[0],
                                     model.config.id2label, image.size)
        ox_dets = detections_to_list(post_process(processor, ox_logits, ox_boxes, target_sizes, **kw)[0],
                                     model.config.id2label, image.size)

        entry = {"image": name, "input_shape": list(inputs["pixel_values"].shape),
                 "max_abs_diff_logits": logits_diff, "max_abs_diff_pred_boxes": boxes_diff,
                 "within_atol": logits_diff <= atol and boxes_diff <= atol}
        entry.update(_match_detections(pt_dets, ox_dets))
        report["images"].append(entry)
        log.info(f"[{name}] shape {entry['input_shape']} | max|Δlogits| {logits_diff:.2e} | max|Δboxes| {boxes_diff:.2e} | "
                 f"detections PyTorch {len(pt_dets)} vs ONNX {len(ox_dets)} | max box Δ {entry['max_box_diff_px']} px")

        if name == "inference_image":
            print_detections(ox_dets, "inference image (ONNX Runtime)")
            draw_detections(image, [d["box"] for d in ox_dets], [d["label"] for d in ox_dets],
                            [d["score"] for d in ox_dets]).save(out_dir / "inference_image_onnx_pred.jpg")
            draw_detections(image, [d["box"] for d in pt_dets], [d["label"] for d in pt_dets],
                            [d["score"] for d in pt_dets]).save(out_dir / "inference_image_pytorch_pred.jpg")
            report["inference_image_detections"] = {"pytorch": pt_dets, "onnx": ox_dets}

    report["all_within_atol"] = all(e["within_atol"] for e in report["images"])
    report["all_detections_match"] = all(e["same_count"] and e["all_matched"] for e in report["images"])
    save_json(report, out_dir / "onnx_vs_pytorch.json")
    status = "PASSED" if report["all_within_atol"] and report["all_detections_match"] else "CHECK REPORT"
    log.info(f"ONNX Runtime vs PyTorch: {status} (atol={atol}) — report: {out_dir / 'onnx_vs_pytorch.json'}")
    return report
