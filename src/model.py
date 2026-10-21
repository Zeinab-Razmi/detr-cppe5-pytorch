"""Model and image-processor construction for the active model."""
from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from .config import Config
from .utils import get_logger

log = get_logger()


def build_image_processor(cfg: Config, source: str | Path | None = None):
    """Image processor of the active model, sized from data.image_size (see Config.processor_kwargs)."""
    source = str(source or cfg.model.name)
    return AutoImageProcessor.from_pretrained(source, **cfg.processor_kwargs())


def build_model(cfg: Config, id2label: dict[int, str], label2id: dict[str, int]) -> torch.nn.Module:
    """Pretrained checkpoint with a new classification head for the CPPE-5 classes."""
    log.info(f"Loading pretrained weights: {cfg.model.name}")
    return AutoModelForObjectDetection.from_pretrained(
        cfg.model.name,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,  # COCO head (91 classes) -> CPPE-5 head (5 classes)
    )


def load_trained(cfg: Config, device: torch.device | None = None):
    """Best fine-tuned weights + their saved image processor (written by training)."""
    path = cfg.best_model_dir
    if not (path / "config.json").exists():
        raise FileNotFoundError(
            f"No trained model in {path}. Run `python main.py train --model {cfg.model_choice}` first."
        )
    model = AutoModelForObjectDetection.from_pretrained(str(path))
    processor = AutoImageProcessor.from_pretrained(str(path))
    if device is not None:
        model.to(device)
    model.eval()
    log.info(f"Loaded best model from {path}")
    return model, processor
