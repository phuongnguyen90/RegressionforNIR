"""Tiện ích dùng chung cho các kịch bản: thêm thư mục gốc vào sys.path, đọc cấu hình
kế thừa."""

from __future__ import annotations

import copy
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import yaml  # noqa: E402


def deep_update(a: dict, b: dict):
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = (
            deep_update(out.get(k, {}), v)
            if isinstance(v, dict) and isinstance(out.get(k), dict)
            else v
        )
    return out


def load_config(path: str, overrides: list[str] | None = None) -> dict:
    with open(path, encoding="utf8") as f:
        cfg = yaml.safe_load(f)
    if "base" in cfg:
        base = load_config(os.path.join(os.path.dirname(path), cfg.pop("base")))
        cfg = deep_update(base, cfg)
    for ov in overrides or []:  # dạng khóa.con=giá_trị, vd. inference.M=1
        key, val = ov.split("=", 1)
        node = cfg
        *parents, last = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[last] = yaml.safe_load(val)
    return cfg


def device():
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def dump(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=float)
