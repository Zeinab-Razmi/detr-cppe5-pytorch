"""Fine-tuning loop. After every epoch the model is evaluated on the validation split and the weights
are saved only when the selection metric improves, so `best_model/` always holds the best epoch."""
from __future__ import annotations

import math
import time

import torch
import yaml
from transformers import get_scheduler

from .config import Config
from .dataset import build_dataloader
from .evaluate import autocast_context, evaluate, format_metrics
from .model import build_image_processor, build_model, load_trained
from .utils import (
    describe_device,
    get_device,
    get_logger,
    move_labels_to_device,
    plot_training_curves,
    save_json,
    set_seed,
)

log = get_logger()


def build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """AdamW without weight decay on biases and normalisation weights (same grouping as HF Trainer)."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}], lr=lr
    )


def is_improvement(value: float, best: float | None, greater_is_better: bool) -> bool:
    if best is None or math.isnan(best):
        return not math.isnan(value)
    return value > best if greater_is_better else value < best


def train(cfg: Config, ds, id2label: dict[int, str], label2id: dict[str, int]) -> dict:
    """Train on ds['train'], select on ds['validation'], keep the best weights in cfg.best_model_dir."""
    set_seed(int(cfg.train.get("seed", cfg.seed)))
    device = get_device()
    use_amp = cfg.model.fp16 and device.type == "cuda"
    log.info(f"Device: {describe_device(device)} | mixed precision (fp16): {use_amp}")

    tc = cfg.train
    metric_name = tc["metric_for_best_model"]
    greater = bool(tc["greater_is_better"])
    patience = tc.get("early_stopping_patience")
    accum = max(int(tc.get("gradient_accumulation_steps", 1)), 1)

    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.raw, f, sort_keys=False)

    processor = build_image_processor(cfg)
    model = build_model(cfg, id2label, label2id).to(device)
    train_loader = build_dataloader(cfg, ds["train"], processor, train=True)
    val_loader = build_dataloader(cfg, ds["validation"], processor, train=False)

    optimizer = build_optimizer(model, float(tc["learning_rate"]), float(tc["weight_decay"]))
    steps_per_epoch = math.ceil(len(train_loader) / accum)
    total_steps = steps_per_epoch * int(tc["epochs"])
    scheduler = get_scheduler(tc["lr_scheduler"], optimizer, num_warmup_steps=int(tc["warmup_steps"]),
                              num_training_steps=total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    log.info(f"Training {cfg.model.name}: {len(ds['train'])} train / {len(ds['validation'])} val images, "
             f"{tc['epochs']} epochs x {steps_per_epoch} steps, selecting best on val '{metric_name}'")

    history: list[dict] = []
    best_value: float | None = None
    best_epoch: int | None = None
    epochs_without_improvement = 0
    global_step = 0

    for epoch in range(1, int(tc["epochs"]) + 1):
        model.train()
        start = time.time()
        running, seen = 0.0, 0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, start=1):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = move_labels_to_device(batch["labels"], device)

            with autocast_context(device, use_amp):
                # like the notebook: pixel_values + labels only (its transform popped pixel_mask)
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss / accum

            loss_value = outputs.loss.detach().item()
            if not math.isfinite(loss_value):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}, step {step}: {loss_value}")

            scaler.scale(loss).backward()
            bs = pixel_values.shape[0]
            running += loss_value * bs
            seen += bs

            if step % accum == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(tc["max_grad_norm"]))
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % int(tc["log_every_n_steps"]) == 0:
                    log.info(f"epoch {epoch} step {global_step}/{total_steps} | loss {running / seen:.4f} | "
                             f"lr {scheduler.get_last_lr()[0]:.2e}")

        train_loss = running / max(seen, 1)
        val = evaluate(model, processor, val_loader, device, id2label,
                       threshold=float(cfg.evaluation["threshold"]),
                       use_nms=bool(cfg.evaluation["use_nms"]) and cfg.model.use_nms,
                       nms_iou=cfg.model.nms_iou, fp16=use_amp,
                       desc=f"val epoch {epoch}")
        if metric_name not in val:
            raise KeyError(f"metric_for_best_model '{metric_name}' not in evaluation results: {sorted(val)}")
        value = float(val[metric_name])

        record = {"epoch": epoch, "train_loss": round(train_loss, 4), "val_loss": val["loss"],
                  "lr": scheduler.get_last_lr()[0], "time_s": round(time.time() - start, 1)}
        record.update({f"val_{k}": v for k, v in val.items()})
        history.append(record)

        improved = is_improvement(value, best_value, greater)
        if improved:
            best_value, best_epoch = value, epoch
            epochs_without_improvement = 0
            model.save_pretrained(cfg.best_model_dir)
            processor.save_pretrained(cfg.best_model_dir)
            save_json({"epoch": epoch, "metric": metric_name, "value": value, "validation_metrics": val},
                      cfg.best_model_dir / "best_metrics.json")
        else:
            epochs_without_improvement += 1

        log.info(f"epoch {epoch}/{tc['epochs']} | train loss {train_loss:.4f} | val loss {val['loss']:.4f} | "
                 f"val {metric_name} {value:.4f} | best {best_value if best_value is not None else float('nan'):.4f} "
                 f"(epoch {best_epoch})"
                 f"{'  ✓ saved best model' if improved else ''} | {record['time_s']}s")

        save_json({"metric": metric_name, "best_epoch": best_epoch, "best_value": best_value, "history": history},
                  out_dir / "training_history.json")
        plot_training_curves(history, metric_name, best_epoch, save_path=cfg.plots_dir / "training_curves.png")

        if patience is not None and epochs_without_improvement >= int(patience):
            log.info(f"Early stopping: no improvement in val {metric_name} for {patience} epochs.")
            break

    if best_epoch is None:
        raise RuntimeError("Validation metric was never finite — no model was saved.")

    plot_training_curves(history, metric_name, best_epoch, save_path=cfg.plots_dir / "training_curves.png",
                         show=bool(cfg.output.get("show_plots", False)))
    log.info(f"Best epoch: {best_epoch} (val {metric_name} = {best_value:.4f}) -> {cfg.best_model_dir}")

    # Reload the saved best weights and confirm they reproduce the recorded validation score.
    best_model, best_processor = load_trained(cfg, device)
    check = evaluate(best_model, best_processor, val_loader, device, id2label,
                     threshold=float(cfg.evaluation["threshold"]),
                     use_nms=bool(cfg.evaluation["use_nms"]) and cfg.model.use_nms,
                     nms_iou=cfg.model.nms_iou, fp16=use_amp,
                     desc="re-check best")
    log.info("Validation metrics of the reloaded best model:\n" + format_metrics(check, "  "))
    return {"best_epoch": best_epoch, "best_value": best_value, "metric": metric_name,
            "validation_metrics": check, "history": history}
