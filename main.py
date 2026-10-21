"""Command-line entry point.

    python main.py analyze                      # data loading, split, distribution plots
    python main.py train                        # analyze + train (best weights by val mAP) + test evaluation
    python main.py evaluate --split test        # mAP of the best weights on a split
    python main.py predict                      # best weights on inference.image (URL or path)
    python main.py predict --image my.jpg
    python main.py predict --split test         # ground truth vs prediction for test images
    python main.py predict --split test --index 3 12   # only test images #3 and #12
    python main.py export                       # ONNX export + ONNX Runtime vs PyTorch comparison
    python main.py all                          # train -> test evaluation -> predict -> export

Common options: --model detr|conditional_detr, --config path/to/config.yaml,
                --set train.epochs=5 data.num_workers=0   (override any config value)
"""
from __future__ import annotations

import argparse
import sys

from src.config import load_config, parse_overrides
from src.utils import get_device, get_logger, save_json, set_seed

log = get_logger()


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=None, help="path to config.yaml (default: project root)")
    common.add_argument("--model", choices=["detr", "conditional_detr"], default=None,
                        help="overrides model_choice from the config")
    common.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="override config values, e.g. --set train.epochs=5 data.num_workers=0")

    parser = argparse.ArgumentParser(description="Fine-tune DETR / Conditional DETR on CPPE-5",
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("analyze", parents=[common], help="load, split and plot the dataset")
    sub.add_parser("train", parents=[common], help="train, keep the best weights, evaluate them on test")
    p_eval = sub.add_parser("evaluate", parents=[common], help="evaluate the best weights")
    p_eval.add_argument("--split", default="test", choices=["train", "validation", "test"])
    p_pred = sub.add_parser("predict", parents=[common], help="run the best weights on an image or a split")
    p_pred.add_argument("--image", default=None, help="image path or URL (default: inference.image)")
    p_pred.add_argument("--split", default=None, choices=["train", "validation", "test"],
                        help="visualise predictions on this split instead of an image")
    p_pred.add_argument("--index", type=int, nargs="+", default=None, metavar="I",
                        help="image index(es) in the split, e.g. --index 3 or --index 0 5 12 (default: inference.test_indices)")
    p_pred.add_argument("--num-images", type=int, default=None, help="first N images when no index is given")
    p_pred.add_argument("--threshold", type=float, default=None, help="default: the model's inference_threshold")
    sub.add_parser("export", parents=[common], help="export to ONNX and compare with PyTorch")
    sub.add_parser("all", parents=[common], help="train -> evaluate test -> predict -> export")
    return parser


def run_evaluation(cfg, ds, id2label, split: str) -> dict:
    from src.dataset import build_dataloader
    from src.evaluate import evaluate, format_metrics
    from src.model import load_trained

    device = get_device()
    model, processor = load_trained(cfg, device)
    loader = build_dataloader(cfg, ds[split], processor, train=False)
    metrics = evaluate(model, processor, loader, device, id2label,
                       threshold=float(cfg.evaluation["threshold"]),
                       use_nms=bool(cfg.evaluation["use_nms"]) and cfg.model.use_nms,
                       nms_iou=cfg.model.nms_iou, fp16=cfg.model.fp16 and device.type == "cuda", desc=split)
    log.info(f"{split} metrics of the best model:\n" + format_metrics(metrics, "  "))
    save_json(metrics, cfg.output_dir / f"{split}_metrics.json")
    return metrics


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, model_choice=args.model, overrides=parse_overrides(args.set))
    set_seed(cfg.seed)
    log.info("Configuration\n" + cfg.summary())

    from src.dataset import analyze_dataset, load_and_split

    # A single image prediction does not need the dataset; every other command does.
    needs_data = not (args.command == "predict" and not args.split)
    if needs_data:
        raw, ds, id2label, label2id = load_and_split(cfg)

    if args.command == "analyze":
        analyze_dataset(cfg, raw, ds, id2label)

    elif args.command in ("train", "all"):
        from src.train import train

        analysis = analyze_dataset(cfg, raw, ds, id2label)
        save_json(analysis, cfg.output_dir / "data_analysis.json")
        train(cfg, ds, id2label, label2id)
        run_evaluation(cfg, ds, id2label, "test")
        if args.command == "all":
            from src.export_onnx import compare_with_pytorch, export_onnx
            from src.inference import predict_image, predict_split

            predict_split(cfg, ds["test"], split_name="test")
            try:
                predict_image(cfg)
            except Exception as exc:
                log.warning(f"Skipping inference.image prediction: {exc}")
            compare_with_pytorch(cfg, export_onnx(cfg), test_split=ds["test"])

    elif args.command == "evaluate":
        run_evaluation(cfg, ds, id2label, args.split)

    elif args.command == "predict":
        from src.inference import predict_image, predict_split

        if args.split:
            predict_split(cfg, ds[args.split], num_images=args.num_images, split_name=args.split,
                          threshold=args.threshold, indices=args.index)
        else:
            predict_image(cfg, args.image, threshold=args.threshold)

    elif args.command == "export":
        from src.export_onnx import compare_with_pytorch, export_onnx

        compare_with_pytorch(cfg, export_onnx(cfg), test_split=ds["test"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
