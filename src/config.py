"""Load config.yaml, apply command-line overrides and resolve the active model."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

SUPPORTED_MODELS = ("detr", "conditional_detr")
RESIZE_MODES = ("fixed_canvas", "edges")


def _set_by_path(d: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = d
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            raise KeyError(f"Unknown config key '{dotted}' (no section '{k}')")
        node = node[k]
    if keys[-1] not in node:
        raise KeyError(f"Unknown config key '{dotted}'")
    node[keys[-1]] = value


def parse_overrides(items: list[str] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must look like key.subkey=value")
        key, raw = item.split("=", 1)
        out[key.strip()] = yaml.safe_load(raw)
    return out


@dataclass
class ModelConfig:
    """Settings of the active model"""
    key: str
    name: str
    inference_threshold: float
    fp16: bool
    use_nms: bool
    nms_iou: float
    resize_mode: str


@dataclass
class Config:
    raw: dict
    path: Path
    model: ModelConfig = field(init=False)

    def __post_init__(self) -> None:
        choice = self.raw.get("model_choice")
        if choice not in SUPPORTED_MODELS:
            raise ValueError(f"Unknown model_choice '{choice}'. Use one of {SUPPORTED_MODELS}.")
        if choice not in self.raw.get("models", {}):
            raise ValueError(f"model_choice '{choice}' has no entry under 'models:' in {self.path}")
        m = self.raw["models"][choice]
        if m.get("resize_mode") not in RESIZE_MODES:
            raise ValueError(f"models.{choice}.resize_mode must be one of {RESIZE_MODES}")
        self.model = ModelConfig(
            key=choice,
            name=m["name"],
            inference_threshold=float(m["inference_threshold"]),
            fp16=bool(m["fp16"]),
            use_nms=bool(m.get("use_nms", False)),
            nms_iou=float(m.get("nms_iou", 0.5)),
            resize_mode=m["resize_mode"],
        )

    # ── section shortcuts ────────────────────────────────────────────────
    @property
    def model_choice(self) -> str:
        return self.model.key

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    @property
    def data(self) -> dict:
        return self.raw["data"]

    @property
    def train(self) -> dict:
        return self.raw["train"]

    @property
    def evaluation(self) -> dict:
        return self.raw["evaluation"]

    @property
    def inference(self) -> dict:
        return self.raw["inference"]

    @property
    def export(self) -> dict:
        return self.raw["export"]

    @property
    def output(self) -> dict:
        return self.raw["output"]

    # ── derived values ──────────────────────────────────────────────────
    def processor_kwargs(self) -> dict:
        """Image-processor arguments, built from data.image_size and the model's resize_mode."""
        s = int(self.data["image_size"])
        if self.model.resize_mode == "fixed_canvas":
            return {
                "do_resize": True,
                "size": {"max_height": s, "max_width": s},
                "do_pad": True,
                "pad_size": {"height": s, "width": s},
            }
        return {"do_resize": True, "size": {"shortest_edge": s, "longest_edge": s}, "do_pad": True}

    @property
    def output_dir(self) -> Path:
        """outputs/detr or outputs/conditional_detr inside the project folder."""
        base = Path(self.output["dir"]).expanduser()
        if not base.is_absolute():
            base = PROJECT_ROOT / base
        return base / self.model_choice

    @property
    def best_model_dir(self) -> Path:
        return self.output_dir / "best_model"

    @property
    def plots_dir(self) -> Path:
        return self.output_dir / "plots"

    @property
    def onnx_path(self) -> Path:
        return self.output_dir / "onnx" / "model.onnx"

    def summary(self) -> str:
        return (
            f"Model        : {self.model.name} ({self.model_choice})\n"
            f"Threshold    : {self.model.inference_threshold}\n"
            f"FP16         : {self.model.fp16}\n"
            f"NMS          : {self.model.use_nms} (iou={self.model.nms_iou})\n"
            f"Image size   : {self.data['image_size']} ({self.model.resize_mode})\n"
            f"Output dir   : {self.output_dir}"
        )


def load_config(
    path: str | Path | None = None,
    model_choice: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> Config:
    """Read config.yaml, optionally switch the model and apply dotted overrides."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    raw = copy.deepcopy(raw)
    if model_choice:
        raw["model_choice"] = model_choice
    for key, value in (overrides or {}).items():
        _set_by_path(raw, key, value)
    return Config(raw=raw, path=path)
