"""CPPE-5 loading, analysis, stratified train/validation split and PyTorch datasets/dataloaders."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from datasets import ClassLabel, DatasetDict, load_dataset, load_from_disk
from torch.utils.data import DataLoader, Dataset

from .config import Config
from .utils import (
    build_transforms,
    collate_fn,
    count_labels,
    draw_sample,
    format_image_annotations_as_coco,
    get_logger,
    plot_class_distribution,
    sanitize_coco_boxes,
)

log = get_logger()


# ════════════════════════════════════════════════════════════════════════
#  Loading & label maps
# ════════════════════════════════════════════════════════════════════════
def load_raw_dataset(cfg: Config) -> DatasetDict:
    """Load from the Hugging Face hub, or from a local folder made with DatasetDict.save_to_disk."""
    source = cfg.data["dataset"]
    if Path(source).expanduser().is_dir():
        log.info(f"Loading dataset from disk: {source}")
        return load_from_disk(str(Path(source).expanduser()))
    log.info(f"Loading dataset from the Hugging Face hub: {source}")
    return load_dataset(source)


def get_label_maps(ds: DatasetDict) -> tuple[dict[int, str], dict[str, int]]:
    categories = ds["train"].features["objects"]["category"].feature.names
    id2label = {i: name for i, name in enumerate(categories)}
    label2id = {name: i for i, name in id2label.items()}
    return id2label, label2id


# ════════════════════════════════════════════════════════════════════════
#  Stratified split
# ════════════════════════════════════════════════════════════════════════
def add_validation_split(ds: DatasetDict, cfg: Config) -> DatasetDict:
    """Create a 'validation' split from 'train', stratified on each image's rarest class.
    """
    if "validation" in ds:
        log.info("Dataset already has a validation split — keeping it.")
        return ds

    names = ds["train"].features["objects"]["category"].feature.names
    priority_names = list(cfg.data["stratify_priority"])
    fallback_name = cfg.data["stratify_fallback"]
    unknown = [n for n in priority_names + [fallback_name] if n not in names]
    if unknown:
        raise ValueError(f"stratify_priority/fallback contain unknown classes {unknown}; dataset classes are {names}")
    missing = [n for n in names if n not in priority_names]
    if missing:
        raise ValueError(f"stratify_priority must list every class; missing {missing}")

    priority_ids = [names.index(n) for n in priority_names]
    fallback_id = names.index(fallback_name)

    def determine_stratify_label(example: dict) -> dict:
        cats = set(example["objects"]["category"])
        for cid in priority_ids:
            if cid in cats:
                return {"stratify_label": cid}
        return {"stratify_label": fallback_id}  # ClassLabel cannot encode -1

    train = ds["train"].map(determine_stratify_label, desc="Assigning stratification labels")
    # Cast AFTER mapping: names are ordered by integer id, so label i -> names[i]
    train = train.cast_column("stratify_label", ClassLabel(names=names))
    split = train.train_test_split(
        test_size=float(cfg.data["val_ratio"]), stratify_by_column="stratify_label", seed=cfg.seed
    )
    new = {"train": split["train"].remove_columns("stratify_label"),
           "validation": split["test"].remove_columns("stratify_label")}
    new.update({k: v for k, v in ds.items() if k != "train"})
    return DatasetDict(new)


def load_and_split(cfg: Config) -> tuple[DatasetDict, DatasetDict, dict[int, str], dict[str, int]]:
    """Returns (raw dataset, dataset with train/validation/test, id2label, label2id)."""
    raw = load_raw_dataset(cfg)
    id2label, label2id = get_label_maps(raw)
    ds = add_validation_split(raw, cfg)
    log.info("Splits: " + ", ".join(f"{k}={len(v)}" for k, v in ds.items()))
    log.info(f"Categories: {id2label}")
    return raw, ds, id2label, label2id


# ════════════════════════════════════════════════════════════════════════
#  Analysis
# ════════════════════════════════════════════════════════════════════════
def analyze_dataset(cfg: Config, raw: DatasetDict, ds: DatasetDict, id2label: dict[int, str], sample_idx: int = 3) -> dict:
    """Class distribution, per-split shares and a sample image."""
    plots = cfg.plots_dir
    show = bool(cfg.output.get("show_plots", False))

    raw_counts = count_labels(raw["train"]["objects"], id2label)
    plot_class_distribution(raw_counts, save_path=plots / "class_distribution.png", show=show)

    split_counts = {name: count_labels(split["objects"], id2label) for name, split in ds.items()}

    idx = min(sample_idx, len(ds["train"]) - 1)
    image, num_boxes, category_ids = draw_sample(ds["train"], idx, id2label)
    sample_path = plots / "sample_train_image.png"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(sample_path)

    images_per_split = {name: len(split) for name, split in ds.items()}
    boxes_per_image = {
        name: float(np.mean([len(o["category"]) for o in split["objects"]])) if len(split) else 0.0
        for name, split in ds.items()
    }
    log.info("Box counts in original train split: " + ", ".join(f"{k}={v}" for k, v in raw_counts.items()))
    for name, counts in split_counts.items():
        total = max(sum(counts.values()), 1)
        shares = ", ".join(f"{k}={100 * v / total:.1f}%" for k, v in counts.items())
        log.info(f"[{name}] images={images_per_split[name]} boxes={total} | {shares}")
    log.info(f"Sample train image #{idx}: {num_boxes} boxes, categories {category_ids}")
    log.info(f"Plots saved to {plots}")
    return {
        "original_train_counts": raw_counts,
        "split_counts": split_counts,
        "images_per_split": images_per_split,
        "mean_boxes_per_image": boxes_per_image,
    }


# ════════════════════════════════════════════════════════════════════════
#  PyTorch dataset & loaders
# ════════════════════════════════════════════════════════════════════════
class CPPE5Dataset(Dataset):
    """Applies albumentations + the model's image processor to one Hugging Face split."""

    def __init__(self, hf_split, image_processor, train: bool) -> None:
        self.split = hf_split
        self.processor = image_processor
        self.transform = build_transforms(train)

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, idx: int) -> dict:
        example = self.split[idx]
        image = example["image"].convert("RGB")
        objects = example["objects"]
        bboxes, categories = sanitize_coco_boxes(objects["bbox"], objects["category"], image.width, image.height)

        out = self.transform(image=np.array(image), bboxes=bboxes, category=categories)
        annotations = format_image_annotations_as_coco(int(example["image_id"]), out["category"], out["bboxes"])

        encoded = self.processor(images=[out["image"]], annotations=[annotations], return_tensors="pt")
        pixel_values = encoded["pixel_values"][0]
        pixel_mask = encoded["pixel_mask"][0] if "pixel_mask" in encoded else torch.ones(pixel_values.shape[-2:], dtype=torch.long)
        return {"pixel_values": pixel_values, "pixel_mask": pixel_mask, "labels": encoded["labels"][0]}


def build_dataloader(cfg: Config, hf_split, image_processor, train: bool) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(cfg.train.get("seed", cfg.seed)))
    return DataLoader(
        CPPE5Dataset(hf_split, image_processor, train=train),
        batch_size=int(cfg.train["batch_size"] if train else cfg.train["eval_batch_size"]),
        shuffle=train,
        num_workers=int(cfg.data["num_workers"]),
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        persistent_workers=int(cfg.data["num_workers"]) > 0,
    )
